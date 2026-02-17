#!/bin/bash
set -euo pipefail

# Condor passes these 6 positional arguments:
ROOT_FILE="$1"
WEIGHT_FILE="$2"
OUT_DIR="$3"
ENTRY_NUMBER="$4"
N_PIXELS="$5"
TAG="$6"

PROJECT=/home/hep/an1522/dark_tridents_wspace
IMG=/vols/sbn/uboone/an1522/docker_image/larcv2_py3_1.1.sif

# Bind the directory containing the ROOT file
DATA_DIR="$(dirname "$ROOT_FILE")"
ROOT_BASENAME="$(basename "$ROOT_FILE")"

# Make sure output dir exists on host (so apptainer can write to it)
mkdir -p "$OUT_DIR"

apptainer exec --nv \
  -B "$PROJECT":/workspace \
  -B "$DATA_DIR":/data \
  "$IMG" /bin/bash -lc "
    cd /workspace/DM-CNN
    source setup_larcv2_dm.sh
    python3 uboone/gradcam_CNN_cli_v2.py \
      --weight-file \"$WEIGHT_FILE\" \
      --input-file  \"/data/$ROOT_BASENAME\" \
      --entry       \"$ENTRY_NUMBER\" \
      --n-pixels    \"$N_PIXELS\" \
      --output-dir  \"$OUT_DIR\" \
      --tag         \"$TAG\" \
      --method gradcam \
      --layer-preset final \
      --diag-curves \
      --normalize
  "


