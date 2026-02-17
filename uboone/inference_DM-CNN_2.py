# Standard python libraries
import os, sys, ROOT
import getopt, time 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

# Pytorch stuff
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

# For rotations
from scipy.ndimage import rotate

# MPID scripts 
from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary

plt.ioff()
torch.cuda.is_available()

from lib.config import config_loader
from lib.utility import get_fname 


def InferenceCNN():
    '''
      Perform inference using a trained DM-CNN model
      the parameters are obtained from a config file 
      returns:
        None 
    '''
    MPID_PATH = os.path.dirname(mpid_data_binary.__file__)+"/../cfg"
    CFG = os.path.join(MPID_PATH,"inference_config_binary_mpid_2.cfg")
    cfg  = config_loader(CFG)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID
    train_device = 'cuda' if torch.cuda.is_available() else 'cpu'


    input_file = cfg.input_file 
    input_csv = cfg.input_csv 

    # Obtain name without extension of the file 
    file_name = get_fname(input_file)

    print("\n")
    print("Running DM-CNN inference...")
    print("Input larcv: "+input_file)
    print("Input csv: "+input_csv)
    print("Rotation: " + str(cfg.rotation))
    print("\n")

    output_dir = cfg.output_dir 
    # create output file name 
    tag = cfg.name 
    output_file = output_dir + file_name + "_DM-CNN_scores_" + tag + ".csv"

    # Weight file 
    weight_file = cfg.weight_file

    # Rotate 
    rotate = cfg.rotation 

    # Input csv 
    df = pd.read_csv(input_csv)
    df['signal_score']=np.ones(len(df))*-999999.9
    df['entry_number']=np.ones(len(df))*-1
    df['n_pixels']=np.ones(len(df))*-1

    # Configure MPID core 
    mpid = mpid_net_binary.MPID()
    mpid.cuda()
    mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
    mpid.eval()


    test_data = mpid_data_binary.MPID_Dataset(input_file,"image2d_image2d_binary_tree", train_device)
    n_events = test_data[0][3]

    print("Total number of events: ", n_events)
    print("Starting...")
    #initialize timer
    init = time.time()
    for ENTRY in range(n_events - 1):
        
        if(ENTRY%1000 == 0):
            print("ENTRY: ", ENTRY)
  
        run_info = test_data[ENTRY][2][0]
        subrun_info = test_data[ENTRY][2][1]
        event_info = test_data[ENTRY][2][2]
        index_array = df.query('run_number == {:2d} & subrun_number == {:2d} & event_number == {:2d} '.format(run_info,subrun_info,event_info)).index.values
        input_image = test_data[ENTRY][0].view(-1,1,512,512)
        input_image[0][0][input_image[0][0] > 500] = 500
        input_image[0][0][input_image[0][0] < 10 ] = 0

        # Image rotation 
        if(rotate):
            input_image[0][0] = torch.tensor(rotate(input_image[0][0],angle=rotation_angle))

        score = nn.Sigmoid()(mpid(input_image.cuda())).cpu().detach().numpy()[0]

        # If the image is not in the csv, skip 
        if(len(index_array) ==0):
            continue
    

        df['signal_score'][index_array[0]]=score[0] 
        df['entry_number'][index_array[0]]=ENTRY
        df['n_pixels'][index_array[0]]=np.count_nonzero(input_image)

    end = time.time()
    print("Total processing time: {:0.4f} seconds".format(end-init))          
    df.to_csv(output_file,index=False)
    plt.figure()

    # Generate score distribution plot 
    # This is a prelminary plot for quick diagnostics 
    # dp=df[df['signal_score'] >= 0.]
    # plt.hist(dp['signal_score'], bins = 40, alpha=0.9, label=file_name,histtype='bar')
    # plt.xlabel("Signal score")
    # plt.grid()
    # plt.legend(loc='upper right')
    # plt.savefig(output_dir + file_name + "_DM-CNN_signal_score_distribution_" + tag + ".png")
    # plt.savefig(output_dir + file_name + "_DM-CNN_signal_score_distribution_" + tag + ".pdf")
    return 0 
 




if __name__ == "__main__":
    InferenceCNN()
