#!/bin/bash

cd /home/hep/an1522/dark_tridents_wspace

apptainer exec --nv -B "$PWD":/workspace /vols/sbn/uboone/an1522/docker_image/larcv2_py3_1.1.sif /bin/bash -lc '
  set -e
  pushd /opt/root >/dev/null
  source bin/thisroot.sh
  popd >/dev/null

  cd /workspace/DM-CNN
  source setup_larcv2_dm.sh
  mkdir -p /workspace/outputs
  mkdir -p /workspace/outputs/weights
  python3 ./uboone/train_DM-CNN.py
'
