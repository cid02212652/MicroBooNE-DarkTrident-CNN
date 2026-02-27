import sys, getopt
# To run this script use the prod_dl_larcv2.sif container 

import os, sys, ROOT  
import getopt, time                                                 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook 
plt.ioff()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary, mpid_func_binary as mpid_func
from lib.config import config_loader
from lib.utility import get_fname, logit_transform
from scipy.ndimage import rotate
from scipy.ndimage import shift
import pandas as pd



def PrintImage(ENTRY):

    '''
    Function that creates event displays of 
    512x512 images that are used with DM-CNN. 
    This function uses the MPID data loader
    to load the images contained in the input file. 
    The images are printed in .png and .pdf format.


    Obligatory arguments:
    ENTRY: entry number of the event to print


    Returns: N/A 

    ''' 



    MPID_PATH = os.path.dirname(mpid_data_binary.__file__)+"/../cfg"
    CFG = os.path.join(MPID_PATH,"print_image_config.cfg")
    cfg  = config_loader(CFG)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPUID
    weight_file = cfg.weight_file
    output_dir = cfg.output_dir  
    input_file = cfg.input_file 
    score_label = cfg.score_label
    colorbar_included = cfg.colorbar
    data = cfg.data 
    logit = cfg.logit 
    logo = cfg.logo

    print("\n")
    print("Processing entry: ",ENTRY)
    
    # Obtain file name without path and without extension 
    file_name = get_fname(input_file)

    # Get image of event  
    train_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Train device: ", train_device)
    # Note: The data product name is hardcoded to "image2d_image2d_binary_tree"
    # You need to change this if your files use a different name 
    test_data = mpid_data_binary.MPID_Dataset(input_file,"image2d_image2d_binary_tree", train_device)
    test_loader = DataLoader(dataset=test_data, batch_size= 1 , shuffle=True)
    input_image = test_data[ENTRY][0].view(-1,1,512,512)
    event_info = test_data[ENTRY][2]
    input_image[0][0][input_image[0][0] > 500] = 500
    input_image[0][0][input_image[0][0] < 10 ] = 0

    # Get scores of input image 
    if(score_label):
        mpid = mpid_net_binary.MPID()
        mpid.load_state_dict(torch.load(weight_file, map_location=train_device))
        mpid.eval()
        score = nn.Sigmoid()(mpid(input_image))

    fig, ax= plt.subplots(1,1,figsize=(20,20),dpi=300)
    img = ax.imshow(input_image.cpu()[0][0].detach().cpu().numpy().transpose(),origin="lower",cmap='jet',norm=colors.PowerNorm(gamma=0.35,vmin=input_image.cpu()[0][0].min(), vmax=input_image.cpu()[0][0].max()))
    

    # Configure optionals 
    if(colorbar_included):
        newax_cb = fig.add_axes([0.0001,0.3,0.2,0.4], anchor='NE', zorder=1)
        cbar = fig.colorbar(img, ax=newax_cb,shrink=0.7,cmap='jet')
        cbar.set_ticks([0.,np.max(input_image.cpu()[0][0].detach().cpu().numpy())])
        cbar.ax.set_yticklabels(['Low charge', 'High charge'],color='white',fontsize=50) 
        newax_cb.axis('off')

    if(logo):
        logo_path = os.path.dirname(mpid_data_binary.__file__)+"/../lib/"
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        im = plt.imread(logo_path + 'kalekologo_noshadow_notrack_inverted.png') # insert local path of the image.
        ax.plot(range(10))
        newax = fig.add_axes([0.15,0.58,0.3,0.3], anchor='NE', zorder=1)
        newax.imshow(im)
        newax.axis('off')

    ax.set_xticks([0, 511])
    ax.set_yticks([511])
    ax.tick_params(axis="y", which='major', direction="out",length=10,width=2.5,pad=10, labelsize=50)
    ax.tick_params(axis="y", which='minor', direction="out",length=10,width=1.0,labelleft=False, labelsize=50)
    ax.tick_params(axis="x", which='major', direction="out",length=10,width=2.5,pad=10, bottom=True,top=False, labelsize=50)
    ax.tick_params(axis="x", which='minor', direction="out",length=10,width=2.0, bottom=True,top=False, labelsize=50)
    ax.set_xlim(0,511)
    ax.set_ylim(0,511)
    ax.set_xlabel('Wire Number', size=55,labelpad=1.0)
    ax.set_ylabel('Drift Time', size=55,labelpad=1.0)


    # More optionals  
    if(data):
        ax.text(20,80, "MicroBooNE NuMI Data", color="white",fontsize=53)
        ax.text(20,20,"Run: {}, Subrun: {}, Event {}".format(event_info[0],event_info[1],event_info[2]),color="white",fontsize=53)

    else:
        ax.text(20,20, "MicroBooNE Simulation", color="white",fontsize=53)


    if(score_label and logit):
        ax.text(20,50, "Signal score: %.3f"%logit_transform(score.cpu().detach().numpy()[0][0]),color="white",fontsize=53)
    elif(score_label and not logit):
        ax.text(20,50, "Signal score: %.3f"%score.cpu().detach().numpy()[0][0],color="white",fontsize=53)
    else:
        pass


    # Create output file name 
    output_name_base = file_name + "_ENTRY_" + str(ENTRY) 
    if(colorbar_included):
        output_name_png = output_name_base +  "_colorbar.png"
        output_name_pdf = output_name_base +  "_colorbar.pdf"
    else:
        output_name_png= output_name_base +  ".png"
        output_name_pdf = output_name_base +  ".pdf"

    plt.savefig(output_dir + output_name_png,bbox_inches="tight")
    plt.savefig(output_dir + output_name_pdf,bbox_inches="tight")

    print("Output file: ",output_name_pdf)



if __name__ == "__main__":
    entry = 0
    entry = None
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"n:")
    except:
        print("Error...")

    opts, args = getopt.getopt(argv, "hc:n:", ["help","config=","entry="])
    for opt, arg in opts:
            if opt in ['-n']: 
                entry = arg
    PrintImage(int(entry))
