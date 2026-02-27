#!/bin/bash

PROJECT=/home/hep/an1522/dark_tridents_wspace
DATA=/vols/sbn/uboone/darkTridents/data/larcv_files/run1_signal
IMG=/vols/sbn/uboone/an1522/docker_image/larcv2_py3_1.1.sif

apptainer exec --nv -B "$PROJECT":/workspace -B "$DATA":/data "$IMG" /bin/bash -lc '
  cd /workspace/DM-CNN
  source setup_larcv2_dm.sh
  python3 ./uboone/occlusion_analysis_CNN.py -n 0

'