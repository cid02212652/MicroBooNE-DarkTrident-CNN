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
from scipy.ndimage import rotate as nd_rotate

# MPID scripts
from mpid_data import mpid_data_binary
from mpid_net import mpid_net_resnet_binary, mpid_net_resnet_binary_better

plt.ioff()
torch.cuda.is_available()

from lib.config import config_loader
from lib.utility import get_fname

def _infer_is_resnet_checkpoint(state_dict):
    """
    Heuristic to detect ResNet wrapper checkpoints (keys like 'net.layer1.0.conv1.weight').
    """
    if not isinstance(state_dict, dict):
        return False
    # Common prefixes in our ResNet wrapper
    for k in state_dict.keys():
        if k.startswith("net.layer") or k.startswith("net.conv1") or k.startswith("net.fc"):
            return True
    return False


def InferenceCNN():
    '''
      Perform inference using a trained ResNet-based model
      the parameters are obtained from a config file
      returns:
        None
    '''
    
    MPID_PATH = os.path.dirname(mpid_data_binary.__file__) + "/../cfg"
    CFG = os.path.join(MPID_PATH, "inference_config_binary_1.cfg")
    cfg = config_loader(CFG)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUID

    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_file = cfg.input_file
    input_csv  = cfg.input_csv

    # Obtain name without extension of the file
    file_name = get_fname(input_file)

    print("\n")
    print("Running ResNet inference...")
    print("Input larcv: " + input_file)
    print("Input csv: " + input_csv)
    print("Rotation: " + str(cfg.rotation))
    print("\n")

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # create output file name
    tag = cfg.name
    output_file = output_dir + file_name + "_ResNet_scores_" + tag + ".csv"

    # Weight file
    weight_file = cfg.weight_file

    # Rotation options
    do_rotate = bool(cfg.rotation)
    rotation_angle = getattr(cfg, "rotation_angle", 90.0)

    # Input csv
    df = pd.read_csv(input_csv)
    df['signal_score']  = np.ones(len(df)) * -999999.9
    df['entry_number']  = np.ones(len(df)) * -1
    df['n_pixels']      = np.ones(len(df)) * -1

    # ---- Build the ResNet model (defaults chosen to be safe if cfg doesn't include them)
    arch           = getattr(cfg, "model", "resnet18")           # e.g. "resnet18" / "resnet34"
    resnet_norm    = getattr(cfg, "resnet_norm", "bn")           # "bn" or "gn"
    gn_groups      = getattr(cfg, "gn_groups", 32)
    drop_out       = getattr(cfg, "drop_out", 0.5)
    pretrained     = getattr(cfg, "resnet_pretrained", False)
    in_channels    = getattr(cfg, "in_channels", 1)
    num_classes    = getattr(cfg, "labels", 2)                   # keep compatible with cfg naming if present

    # If labels isn't in cfg, force the binary setup you trained with
    if num_classes is None:
        num_classes = 2

    mpid = mpid_net_resnet_binary_better.MPID(
        arch=arch,
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=drop_out,
        pretrained=pretrained,
        norm=resnet_norm,
        gn_groups=gn_groups,
    )

    mpid.to(train_device)

    # Load checkpoint
    state = torch.load(weight_file, map_location=train_device)

    # Some people save full dicts; handle both cases gracefully
    if isinstance(state, dict) and ("model_state_dict" in state):
        state = state["model_state_dict"]

    # If someone accidentally points to a non-resnet checkpoint, error loudly with a helpful message
    if not _infer_is_resnet_checkpoint(state):
        raise RuntimeError(
            "This looks like a non-ResNet checkpoint (missing 'net.layer*/net.conv1/net.fc' keys). "
            "Make sure cfg.weight_file points to a ResNet .pwf and you're using inference_DM-CNN_resnet.py."
        )

    mpid.load_state_dict(state, strict=True)
    mpid.eval()

    # Data
    test_data = mpid_data_binary.MPID_Dataset(input_file, "image2d_image2d_binary_tree", train_device)
    n_events = test_data[0][3]

    print("Total number of events: ", n_events)
    print("Starting...")

    init = time.time()

    for ENTRY in range(n_events - 1):

        if (ENTRY % 1000 == 0):
            print("ENTRY: ", ENTRY)

        run_info    = test_data[ENTRY][2][0]
        subrun_info = test_data[ENTRY][2][1]
        event_info  = test_data[ENTRY][2][2]

        index_array = df.query(
            'run_number == {:2d} & subrun_number == {:2d} & event_number == {:2d} '.format(
                run_info, subrun_info, event_info
            )
        ).index.values

        input_image = test_data[ENTRY][0].view(-1, 1, 512, 512)

        # Clamp values (as in your original script)
        input_image[0][0][input_image[0][0] > 500] = 500
        input_image[0][0][input_image[0][0] < 10 ] = 0

        # Image rotation
        if do_rotate:
            img_np = input_image[0][0].cpu().numpy()
            img_rot = nd_rotate(img_np, angle=rotation_angle, reshape=False, mode="nearest")
            input_image[0][0] = torch.tensor(img_rot, dtype=input_image.dtype)

        # Forward
        with torch.no_grad():
            logits = mpid(input_image.to(train_device)).detach().cpu().numpy()[0]

        # Binary setup: if 1 logit -> sigmoid scalar; if 2 logits -> sigmoid per-class and take [0] like old script
        if np.ndim(logits) == 0 or (hasattr(logits, "shape") and logits.shape == ()):
            score_val = float(1.0 / (1.0 + np.exp(-logits)))
        else:
            probs = 1.0 / (1.0 + np.exp(-logits))
            score_val = float(probs[0])

        # If the image is not in the csv, skip
        if (len(index_array) == 0):
            continue

        df['signal_score'][index_array[0]] = score_val
        df['entry_number'][index_array[0]] = ENTRY
        df['n_pixels'][index_array[0]] = np.count_nonzero(input_image.numpy())

    end = time.time()
    print("Total processing time: {:0.4f} seconds".format(end - init))

    df.to_csv(output_file, index=False)

    # Generate score distribution plot
    # dp = df[df['signal_score'] >= 0.]
    # plt.figure()
    # plt.hist(dp['signal_score'], bins=40, alpha=0.9, label=file_name, histtype='bar')
    # plt.xlabel("Signal score")
    # plt.grid()
    # plt.legend(loc='upper right')
    # plt.savefig(output_dir + file_name + "_ResNet_signal_score_distribution_" + tag + ".png")
    # plt.savefig(output_dir + file_name + "_ResNet_signal_score_distribution_" + tag + ".pdf")

    return 0


if __name__ == "__main__":
    InferenceCNN()

