import os, sys, ROOT
import getopt, time 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

# MPID scripts 
from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary
from lib.config import config_loader
from lib.utility import get_fname


plt.ioff()
torch.cuda.is_available()



def add_mask(input_tensor):
    input_tensor[input_tensor>0] = 0 
    return input_tensor



def score_plot(score_map,output_dir, tag, title,vmin, vmax, cmap_input="gnuplot_r"):

    '''
    This function creates an occlusion score map
    and prints it into png and pdf files.

    It normalizes the input map provided the minimum 
    and maximum scores obtained for the input image


    Parameters: 

    score_mape: torch tensor containing the map score
    output_dir: output_directory to store the maps
    tag: name of the input file
    title: string declaring if signal or background map
    vmin: minimum score of the map
    vmax: maximum score of the map
    cmap_input = matplotlib colormap 

    '''

    fig, ax = plt.subplots(1,1,figsize=(20,20),dpi=200)
    pos = ax.imshow(score_map.transpose(),origin="lower", cmap=cmap_input, vmin=vmin, vmax=vmax)     
    ax.set_xlabel("%s Score Map"%title, fontsize=35,labelpad=20)
    cbar = fig.colorbar(pos, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)

    plt.savefig(output_dir + "occlusion_test_{}_{}_map.png".format(tag,title),bbox_inches="tight")
    plt.savefig(output_dir + "occlusion_test_{}_{}_map.pdf".format(tag,title),bbox_inches="tight")

    print("Output file: " + "occlusion_test_{}_{}_map.pdf".format(tag,title))


def RunOcclusion(input_entry):
    MPID_PATH = os.path.dirname(mpid_data_binary.__file__)+"/../cfg"
    CFG = os.path.join(MPID_PATH,"occlusion_config.cfg")
    cfg  = config_loader(CFG)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID
    input_file = cfg.input_file
    output_dir = cfg.output_dir
    occlusion_size = cfg.occlusion_size
    normalized=cfg.normalization 
    weight_file=cfg.weight_file 
    print("\n")

    # Obtain file name without path and without extension 
    file_name = get_fname(input_file)


    train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configure MPID core 
    mpid = mpid_net_binary.MPID()
    mpid.cuda()
    mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
    mpid.eval()

    # Load dataset 
    test_data = mpid_data_binary.MPID_Dataset(input_file,"image2d_image2d_binary_tree", train_device)
    test_loader = DataLoader(dataset=test_data, batch_size= 1 , shuffle=True)

    # Scanning occlusion 
    occlusion_step = occlusion_size
    entry_start=input_entry 
    entries=1

    for ENTRY in range(entry_start, entry_start + entries):
        input_image = test_data[ENTRY][0].view(-1,1,512,512)
        input_image[0][0][input_image[0][0] > 500] = 500
        input_image[0][0][input_image[0][0] < 10 ] = 0
        
        score = nn.Sigmoid()(mpid(input_image.cuda()))
    
        score_map_signal = np.full([512-occlusion_step, 512-occlusion_step], score.cpu().detach().numpy()[0][0])
        score_map_background = np.full([512-occlusion_step, 512-occlusion_step], score.cpu().detach().numpy()[0][1])

        clone_image = input_image.cpu().clone()
        
        for x in range(0 + occlusion_step, 512 - occlusion_step):
            for y in range(0 + occlusion_step,512 - occlusion_step):
                clone_image = input_image.cpu().clone()

                if(clone_image[0][0][x,y]==0):
                    continue
                

                clone_image[0][0][x-occlusion_step:x+occlusion_step+1,
                                y-occlusion_step:y+occlusion_step+1] = torch.zeros([2*occlusion_step+1, 2*occlusion_step+1])
                
                score = nn.Sigmoid()(mpid(clone_image.cuda())).cpu().detach().numpy()[0]
                score_map_signal[x,y] = score[0]
                score_map_background[x,y] = score[1]
    

    # Make plots 

    if(normalized):
        vmin_signal = np.min(score_map_signal)
        vmax_signal = np.max(score_map_signal)

        delta = vmax_signal - vmin_signal
        score_map_signal_norm = (score_map_signal - vmin_signal)/(delta)


        vmin_background = np.min(score_map_background)
        vmax_background = np.max(score_map_background)
        delta_background = vmax_background - vmin_background
        score_map_background_norm = (score_map_background - vmin_background)/(delta_background)

        vmin_signal_final = 0.
        vmax_signal_final = 1.0

        vmin_background_final = 0.
        vmax_background_final = 1.0
    else:
        score_map_signal_norm = score_map_signal 
        vmin_signal_final = np.min(score_map_signal)
        vmax_signal_final  = np.max(score_map_signal)
        
        score_map_background_norm = score_map_background
        vmin_background_final = np.min(score_map_background)
        vmax_background_final  = np.max(score_map_background)
        

    image_tag = file_name + "_ENTRY_{}".format(input_entry)
    score_plot(score_map_signal_norm, output_dir, image_tag,"Signal",vmin_signal_final,vmax_signal_final,cmap_input='gnuplot_r')
    score_plot(score_map_background_norm,output_dir, image_tag, "Background",vmin_background_final,vmax_background_final)


if __name__ == "__main__":
    entry = None
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"n:")
    except:
        print("Error...")

    for opt, arg in opts:
            if opt in ['-n']: 
                entry = arg
    RunOcclusion(int(entry))
