#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import importlib.util

import torch
import torch.nn as nn

from collections import OrderedDict

def import_from_path(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------- Pretty wrappers ----------------

class PrettyResNet(nn.Module):
    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.stem = nn.Sequential(OrderedDict([
            ("conv1", resnet.conv1),
            ("norm1", resnet.bn1 if hasattr(resnet, "bn1") else nn.Identity()),
            ("relu",  resnet.relu),
            ("pool",  resnet.maxpool if hasattr(resnet, "maxpool") else nn.Identity()),
        ]))
        self.stage1 = resnet.layer1
        self.stage2 = resnet.layer2
        self.stage3 = resnet.layer3
        self.stage4 = resnet.layer4
        self.head = nn.Sequential(OrderedDict([
            ("avgpool", resnet.avgpool),
            ("flatten", nn.Flatten(1)),
            ("fc",      resnet.fc),
        ]))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x


class PrettyMPIDBinary(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.features = getattr(m, "features", nn.Identity())
        self.dropout = getattr(m, "dropout", nn.Identity())
        self.flatten = nn.Flatten(1)  # <-- KEY FIX
        self.classifier = getattr(m, "classifier", nn.Identity())

        # Only keep sigmoid here if your classifier does NOT already include it
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = self.flatten(x)          # <-- KEY FIX
        x = self.classifier(x)
        return self.sigmoid(x)       # remove this if your classifier already has sigmoid


# ---------------- Export ----------------

def export_onnx(model: nn.Module, x: torch.Tensor, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        x,
        str(outpath),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,   # simplest; fixed 1×C×H×W export is fine for viz
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

    # --- ResNets ---
    resnet_mod = import_from_path(Path(args.resnet_file), "mpid_resnet_mod")
    MPIDResNet = resnet_mod.MPID

    for arch, norm in [("resnet18","bn"), ("resnet18","gn"), ("resnet34","bn"), ("resnet34","gn")]:
        wrapper = MPIDResNet(arch=arch, norm=norm, in_channels=args.C, pretrained=False).eval()

        # unwrap to the torchvision model if it exists
        core = getattr(wrapper, "net", wrapper)
        pretty = PrettyResNet(core)

        export_onnx(pretty, x, outdir / f"{arch}_{norm}_pretty.onnx")

    # --- MPID binary ---
    mpid_mod = import_from_path(Path(args.mpid_file), "mpid_binary_mod")
    MPIDBinary = mpid_mod.MPID
    m = MPIDBinary().eval()
    pretty = PrettyMPIDBinary(m)
    export_onnx(pretty, x, outdir / "mpid_binary_pretty.onnx")


if __name__ == "__main__":
    main()
