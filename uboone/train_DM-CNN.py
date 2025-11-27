from __future__ import division
from __future__ import print_function


# TO DO: IMPLEMENT MULTIPLE GPUs 

import torch
# We set this here, otherwise pytorch can't recognize CUDA
print("Checking if CUDA is availbale: ")
print(torch.cuda.is_available())
print("\n")

# Torch utils 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from torchvision import transforms


import os, sys, getopt 
from lib.config import config_loader
from lib.utility import timestr
import numpy as np
import time 

# import MPID core 
from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary, mpid_func_binary



def TrainCNN():
    '''
    Function that trains the DM-CNN model
    It stores the values of the trainig metrics
    in a csv file that will be created in output_dir.
    The weights of the CNN will be stored in weights_dir

    The input files, output directories and settings for this 
    function are given in a config file

    '''

    # Get config file 
    print("Reading config file...\n")
    BASE_PATH = os.path.realpath(__file__)
    BASE_PATH = os.path.dirname(BASE_PATH)
    CFG = os.path.join(BASE_PATH,"../cfg","training_config.cfg")
    cfg  = config_loader(CFG)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID

    # Get configs
    title = cfg.name
    output_dir = cfg.output_directory
    train_file = cfg.input_train
    test_file = cfg.input_test 
    weights_dir = cfg.weights_directory 
    SEED = cfg.seed_number 

    print("Inputs given: ")
    print("Training file: ", train_file)
    print("Test file: ", test_file)
    print("Output directory: ", output_dir)
    print("Weights directory: ", weights_dir)
    print("Seed?: ", cfg.seed)
    print("\n")



    # Create file to store training metrics 
    fout = open(output_dir + 'DM-CNN_training_metrics_{}_{}.csv'.format(timestr(), title), 'w')
    fout.write('train_accu,test_accu,train_loss,test_loss,epoch,step')
    fout.write('\n')

    # String used to create files that will contain the CNN weights
    CNN_weights = weights_dir + "DM-CNN_model_{}_epoch_{}_batch_id_{}_labels_{}_title_{}_step_{}.pwf"

    cuda = torch.cuda.is_available()

    # For reproducibility
    if(cfg.seed and cuda):
        print("Using seed number: ", SEED)
        torch.cuda.manual_seed(SEED)
    elif(cfg.seed and not cuda):
        print("Using seed number: ", SEED)
        torch.manual_seed(SEED)
    else:
        print("A seed has not been defined...")
        print("\n")

    print ("There are {} GPUs available".format(torch.cuda.device_count()))
    train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Note: "image2d_image2d_binary_tree" is the name of the data product stored in the LArCV-ROOT file
    # If your images are labeled under a different data product name change this accordingly. 

    # Training data
    train_data = mpid_data_binary.MPID_Dataset(train_file, "image2d_image2d_binary_tree", train_device, plane=0, augment=False)
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size_train, shuffle=True)
    labels = 2

    # Test data
    test_data = mpid_data_binary.MPID_Dataset(test_file, "image2d_image2d_binary_tree", train_device, plane=0)
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size_test, shuffle=True)

    # Import the CNN model 
    mpid = mpid_net_binary.MPID(dropout=cfg.drop_out, num_classes=2)
    mpid.cuda()

    # Using BCEWithLogitsLoss instead of 
    # Using Sigmoid in mpidnet + BCELoss 
    loss_fn = nn.BCEWithLogitsLoss()


    optimizer  = optim.Adam(mpid.parameters(), lr=cfg.learning_rate)#, weight_decay=0.001)
    train_step = mpid_func_binary.make_train_step(mpid, loss_fn, optimizer)
    test_step  = mpid_func_binary.make_test_step(mpid, test_loader, loss_fn, optimizer)

    print ("Training with {} images".format(len(train_loader.dataset)))

    train_losses = []
    train_accuracies =[]
    test_losses = []
    test_accuracies =[]

    EPOCHS = cfg.EPOCHS
    print ("Start DM-CNN training...")

    step=0

    #initialize timer
    init = time.time()
    for epoch in range(EPOCHS):
        print ("\n")
        print (" @{}th epoch...".format(epoch))
        for batch_idx, (x_batch, y_batch, info_batch, nevents_batch) in enumerate(train_loader):
            print ("\n")
            # the dataset "lives" in the CPU, so do our mini-batches
            # therefore, we need to send those mini-batches to the
            # device where the model "lives"
            print (" @{}th epoch, @ batch_id {}".format(epoch, batch_idx))
            
            x_batch = x_batch.to(train_device).view((-1,1,512,512))
            y_batch = y_batch.to(train_device)   
            loss = train_step(x_batch, y_batch) #model.train() called in train_step
            train_losses.append(loss)
            print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                EPOCHS-1,
                batch_idx * len(x_batch), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss), 
                end='')
            if (batch_idx % cfg.test_every_step == 1 and cfg.run_test):
                if (cfg.save_weights and epoch >= 3 and epoch <= 6):
                    torch.save(mpid.state_dict(), CNN_weights.format(timestr(), epoch, batch_idx,labels, title, step))

                print ("Start eval on test sample.......@step..{}..@epoch..{}..@batch..{}".format(step,epoch, batch_idx))
                test_accuracy = mpid_func_binary.validation(mpid, test_loader, cfg.batch_size_test, train_device, event_nums=cfg.test_events_nums)
                print ("Test Accuracy {}".format(test_accuracy))
                print ("Start eval on training sample...@epoch..{}.@batch..{}".format(epoch, batch_idx))
                train_accuracy = mpid_func_binary.validation(mpid, train_loader, cfg.batch_size_train, train_device, event_nums=cfg.test_events_nums)
                print ("Train Accuracy {}".format(train_accuracy))
                test_loss= test_step(test_loader, train_device)
                print ("Test Loss {}".format(test_loss))
                fout.write("%f,"%train_accuracy)        
                fout.write("%f,"%test_accuracy)
                fout.write("%f,"%loss)
                fout.write("%f,"%test_loss)
                fout.write("%f,"%epoch)
                fout.write("%f"%step)
                fout.write("\n")
            step+=1
    fout.close()
    # end timer
    end = time.time()
    print("\n")
    print("Total training time: {:0.4f} seconds".format(end-init))
    return 0 



if __name__ == '__main__':
    TrainCNN()

