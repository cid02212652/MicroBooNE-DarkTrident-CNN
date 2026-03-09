#!/bin/bash
set -e

cd /home/hep/an1522/dark_tridents_wspace

# Reduce OpenMP/BLAS thread craziness (often helps ROOT+torch+extensions stability)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
export PYTORCH_NVML_BASED_CUDA_CHECK=0

apptainer exec -B "$PWD":/workspace /vols/sbn/uboone/an1522/larcv2_py3_1.1_sparseconvnet.sif /bin/bash -lc '
  set -e

  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export PYTHONFAULTHANDLER=1
  export PYTORCH_NVML_BASED_CUDA_CHECK=0

  # Preflight (prints in the job stdout so you know what node you landed on)
  python3 - << "PY"
import torch
print("torch:", torch.__version__, "torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

  # Constrain torch threads explicitly
  python3 - << "PY"
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print("torch threads set OK")
PY

  # ROOT
  pushd /opt/root >/dev/null
  source bin/thisroot.sh
  popd >/dev/null

  cd /workspace/DM-CNN
  source setup_larcv2_dm.sh
  mkdir -p /workspace/outputs/weights

  python3 ./uboone/train_DM-CNN_sparse_net.py
'