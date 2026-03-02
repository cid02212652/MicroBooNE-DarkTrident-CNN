#!/bin/bash
set -euo pipefail

PROJECT=/home/hep/an1522/dark_tridents_wspace
IMG=/vols/sbn/uboone/an1522/docker_image/larcv2_py3_1.1.sif
OUTDIR="$PROJECT/outputs/network_visualisation/"
mkdir -p "$OUTDIR"

apptainer exec -B "$PROJECT":/workspace "$IMG" /bin/bash -lc "
    set -euo pipefail
    
    cd /workspace/DM-CNN

    export CUDA_VISIBLE_DEVICES=""
    python3 -c 'import torch; import torchvision; print(torch.__version__, torchvision.__version__)'
    
    python3 /workspace/DM-CNN/visualise_network_pretty.py \
      --outdir \"$OUTDIR\" \
      --resnet-file /workspace/DM-CNN/mpid_net/mpid_net_resnet_binary_better.py \
      --mpid-file   /workspace/DM-CNN/mpid_net/mpid_net_binary.py \
      --H 512 --W 512 --C 1
  "

