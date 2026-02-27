#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import importlib.util

import torch

def import_from_path(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def export(model, x, outpath: Path, output_name="logits"):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        x,
        str(outpath),
        opset_version=17,
        input_names=["input"],
        output_names=[output_name],
        dynamic_axes={"input": {0: "batch"}, output_name: {0: "batch"}},
    )
    print("[OK]", outpath)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--resnet-file", required=True)
    ap.add_argument("--mpid-file", required=True)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--C", type=int, default=1)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    x = torch.zeros((1, args.C, args.H, args.W), dtype=torch.float32)

    # ResNets
    resnet_mod = import_from_path(Path(args.resnet_file), "mpid_resnet_mod")
    MPIDResNet = resnet_mod.MPID

    for arch, norm in [("resnet18","bn"), ("resnet18","gn"), ("resnet34","bn"), ("resnet34","gn")]:
        m = MPIDResNet(arch=arch, norm=norm, in_channels=args.C, pretrained=False)
        export(m, x, outdir / f"{arch}_{norm}.onnx")

    # MPID binary
    mpid_mod = import_from_path(Path(args.mpid_file), "mpid_binary_mod")
    MPIDBinary = mpid_mod.MPID
    m = MPIDBinary()
    export(m, x, outdir / "mpid_binary.onnx")

if __name__ == "__main__":
    main()
