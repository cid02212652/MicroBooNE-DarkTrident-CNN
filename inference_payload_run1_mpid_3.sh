#!/bin/bash

cd /home/hep/an1522/dark_tridents_wspace

apptainer exec --nv -B "$PWD":/workspace -B "/vols:/vols" /vols/sbn/uboone/an1522/docker_image/larcv2_py3_1.1.sif /bin/bash -lc '
    cd /workspace/DM-CNN
    source setup_larcv2_dm.sh
    python3 ./uboone/inference_DM-CNN_3.py
'
