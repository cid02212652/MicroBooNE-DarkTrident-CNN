
from __future__ import division
from __future__ import print_function

# train_DM-CNN_sparseSSnet.py
# SparseSSNet-style event classifier training script, matching your existing trainers.  

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.config import config_loader
from lib.utility import timestr

# dataset (LArCV ROOT -> dense 512x512 torch tensor) 
from mpid_data import mpid_data_binary

# training helpers (train_step/test_step/validation) used everywhere else 
from mpid_net import mpid_func_binary
from mpid_net import sparse_net_binary


def TrainCNN():
    print("Checking if CUDA is available: ")
    print(torch.cuda.is_available())
    print("\n")

    # Get config file
    print("Reading config file...\n")
    BASE_PATH = os.path.realpath(__file__)
    BASE_PATH = os.path.dirname(BASE_PATH)
    CFG = os.path.join(BASE_PATH, "../cfg", "training_config_sparse_net.cfg")
    cfg = config_loader(CFG)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUID

    # Get configs
    output_dir = cfg.output_directory
    train_file = cfg.input_train
    test_file = cfg.input_test
    weights_dir = cfg.weights_directory
    SEED = cfg.seed_number

    model_name = "sparsessnet"

    print("Inputs given: ")
    print("Training file: ", train_file)
    print("Test file: ", test_file)
    print("Output directory: ", output_dir)
    print("Weights directory: ", weights_dir)
    print("Seed?: ", cfg.seed)
    print("\n")

    # Create file to store training metrics
    fout = open(
        output_dir + "{}_training_metrics_{}.csv".format(model_name, timestr()),
        "w",
    )
    fout.write("train_accu,test_accu,train_loss,test_loss,epoch,step\n")

    # String used to create files that will contain the CNN weights
    CNN_weights = weights_dir + "{}_model_{}_epoch_{}_batch_id_{}_labels_{}_step_{}.pwf"

    cuda = torch.cuda.is_available()

    # For reproducibility
    if cfg.seed and cuda:
        print("Using seed number: ", SEED)
        torch.cuda.manual_seed(SEED)
    elif cfg.seed and not cuda:
        print("Using seed number: ", SEED)
        torch.manual_seed(SEED)
    else:
        print("A seed has not been defined...")
        print("\n")

    print("There are {} GPUs available".format(torch.cuda.device_count()))
    # train_device = "cuda" if torch.cuda.is_available() else "cpu"
    train_device = "cpu"

    # Data
    train_data = mpid_data_binary.MPID_Dataset(
        train_file,
        "image2d_image2d_binary_tree",
        train_device,
        plane=getattr(cfg, "plane", 0),
        augment=getattr(cfg, "augment", False),
    )
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size_train, shuffle=True)
    labels = 2

    test_data = mpid_data_binary.MPID_Dataset(
        test_file,
        "image2d_image2d_binary_tree",
        train_device,
        plane=getattr(cfg, "plane", 0),
    )
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size_test, shuffle=True)

    # Model (SparseSSNet-style sparse UResNet + event pooling)
    mpid = sparse_net_binary.MPID(
        dropout=getattr(cfg, "drop_out", 0.0),
        num_classes=2,
        spatial_size=getattr(cfg, "spatial_size", 512),
        data_dim=getattr(cfg, "data_dim", 2),
        uresnet_num_strides=getattr(cfg, "sparse_uresnet_num_strides", getattr(cfg, "uresnet_num_strides", 5)),
        uresnet_filters=getattr(cfg, "sparse_uresnet_filters", getattr(cfg, "uresnet_filters", 16)),
        reps=getattr(cfg, "sparse_reps", 2),
        pool=getattr(cfg, "sparse_pool", "meanmax"),
        dense_threshold=getattr(cfg, "sparse_dense_threshold", 0.0),
    ).to(train_device)

    # Loss/optim: stay consistent with binary setup 
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(mpid.parameters(), lr=cfg.learning_rate)

    train_step = mpid_func_binary.make_train_step(mpid, loss_fn, optimizer)
    test_step = mpid_func_binary.make_test_step(mpid, test_loader, loss_fn, optimizer)

    print("Training with {} images".format(len(train_loader.dataset)))

    EPOCHS = cfg.EPOCHS
    print("Start SparseSSNet-style training...")

    step = 0
    init = time.time()
    for epoch in range(EPOCHS):
        print("\n @{}th epoch...".format(epoch))
        for batch_idx, (x_batch, y_batch, info_batch, nevents_batch) in enumerate(train_loader):
            print("\n @{}th epoch, @ batch_id {}".format(epoch, batch_idx))

            # match existing trainer input shape 
            x_batch = x_batch.to(train_device).view((-1, 1, 512, 512))
            y_batch = y_batch.to(train_device)

            loss = train_step(x_batch, y_batch)
            print(
                "\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    EPOCHS - 1,
                    batch_idx * len(x_batch),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss,
                ),
                end="",
            )

            if batch_idx % cfg.test_every_step == 1 and cfg.run_test:
                if cfg.save_weights and epoch >= 4 and epoch <= 6:
                    torch.save(
                        mpid.state_dict(),
                        CNN_weights.format(model_name, timestr(), epoch, batch_idx, labels, step),
                    )

                print(
                    "Start eval on test sample.......@step..{}..@epoch..{}..@batch..{}".format(
                        step, epoch, batch_idx
                    )
                )
                test_accuracy = mpid_func_binary.validation(
                    mpid,
                    test_loader,
                    cfg.batch_size_test,
                    train_device,
                    event_nums=cfg.test_events_nums,
                )
                print("Test Accuracy {}".format(test_accuracy))

                print("Start eval on training sample...@epoch..{}.@batch..{}".format(epoch, batch_idx))
                train_accuracy = mpid_func_binary.validation(
                    mpid,
                    train_loader,
                    cfg.batch_size_train,
                    train_device,
                    event_nums=cfg.test_events_nums,
                )
                print("Train Accuracy {}".format(train_accuracy))

                test_loss = test_step(test_loader, train_device)
                print("Test Loss {}".format(test_loss))

                fout.write("%f,%f,%f,%f,%f,%f\n" % (train_accuracy, test_accuracy, loss, test_loss, epoch, step))

            step += 1

    fout.close()
    end = time.time()
    print("\nTotal training time: {:0.4f} seconds".format(end - init))
    return 0


if __name__ == "__main__":
    TrainCNN()
