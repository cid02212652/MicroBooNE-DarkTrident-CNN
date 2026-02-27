#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Matplotlib only for saving plots (Agg backend safe on batch)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# -----------------------------
# Helpers: checkpoint + model detection
# (mirrors your occlusion script’s robust loading behavior)
# -----------------------------
def _extract_state(ckpt):
    # support: raw state_dict, {"state_dict": ...}, {"model_state_dict": ...}
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # might already be a state_dict
        return ckpt
    raise ValueError("Unsupported checkpoint format (expected dict / state_dict).")


def _looks_like_resnet_state(state: Dict[str, torch.Tensor]) -> bool:
    keys = list(state.keys())
    return (
        any(k.startswith("net.layer") or ".layer" in k for k in keys)
        or any(k.startswith("layer") for k in keys)
    )


def _infer_resnet_depth(state: Dict[str, torch.Tensor]) -> int:
    keys = list(state.keys())
    for k in keys:
        if ".layer1.2." in k or "layer1.2." in k:
            return 34
        if ".layer2.2." in k or "layer2.2." in k:
            return 34
    return 18


def _infer_norm(state: Dict[str, torch.Tensor]) -> str:
    keys = list(state.keys())
    if any(k.endswith("running_mean") or k.endswith("running_var") for k in keys):
        return "bn"
    return "gn"


def _summarise_ckpt(state: Dict[str, torch.Tensor]) -> str:
    if not _looks_like_resnet_state(state):
        return "mpid"
    depth = _infer_resnet_depth(state)
    norm = _infer_norm(state)
    return f"resnet{depth}_{norm}"


# -----------------------------
# Models
# -----------------------------
class MPIDBinary(nn.Module):
    def __init__(self):
        from mpid_net import mpid_net_binary
        super().__init__()
        self.net = mpid_net_binary.MPID()

    def forward(self, x):
        return self.net(x)


def _replace_bn_with_gn(module: nn.Module, num_groups: int = 32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=child.num_features,
                eps=child.eps,
                affine=True,
            )
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, num_groups=num_groups)


class ResNetBinaryWrapper(nn.Module):
    def __init__(self, depth: int = 18, norm: str = "bn", dropout: float = 0.0):
        super().__init__()
        from torchvision.models import resnet18, resnet34

        if depth == 34:
            net = resnet34(weights=None)
        else:
            net = resnet18(weights=None)

        # 1-channel input
        net.conv1 = nn.Conv2d(
            1, net.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )

        if norm == "gn":
            _replace_bn_with_gn(net)

        # 2 logits (signal/background)
        in_features = net.fc.in_features
        net.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 2))
        self.net = net

    def forward(self, x):
        return self.net(x)


def _try_load(model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> Optional[str]:
    try:
        model.to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return None
    except Exception as e:
        return str(e)


def build_model(model_key: str, weight_file: str, device: torch.device) -> Tuple[nn.Module, str]:
    """
    Robust loader:
      - tries requested model_key first
      - else guesses from checkpoint keys
      - else brute-force resnet variants if needed
    Returns: (model, resolved_model_key)
    """
    model_key = (model_key or "auto").lower()
    ckpt = torch.load(weight_file, map_location=device)
    state = _extract_state(ckpt)
    guessed = _summarise_ckpt(state)

    def make(key: str) -> nn.Module:
        if key in ["mpid", "dmcnn", "dm-cnn"]:
            return MPIDBinary().net
        if key.startswith("resnet"):
            depth = 34 if "34" in key else 18
            norm = "gn" if key.endswith("_gn") else "bn"
            return ResNetBinaryWrapper(depth=depth, norm=norm, dropout=0.0)
        raise ValueError(f"Unknown model key: {key}")

    # 1) requested
    if model_key != "auto":
        model = make(model_key)
        err = _try_load(model, state, device)
        if err is None:
            return model, model_key

    # 2) guessed
    model = make(guessed)
    err = _try_load(model, state, device)
    if err is None:
        return model, guessed

    # 3) brute-force for resnet-like
    if _looks_like_resnet_state(state):
        for c in ["resnet18_bn", "resnet18_gn", "resnet34_bn", "resnet34_gn"]:
            model = make(c)
            err2 = _try_load(model, state, device)
            if err2 is None:
                return model, c

    raise RuntimeError(
        "Failed to load weights.\n"
        f"Requested={model_key}, Guessed={guessed}\n"
        f"Last error:\n{err}"
    )


# -----------------------------
# Plotting + IO (matches your occlusion script style)
# -----------------------------
def clamp_adc(img: torch.Tensor, adc_lo: float, adc_hi: float) -> torch.Tensor:
    img = img.clone()
    img[img > adc_hi] = adc_hi
    img[img < adc_lo] = 0.0
    return img


def norm01(a: np.ndarray) -> np.ndarray:
    amin, amax = float(np.min(a)), float(np.max(a))
    if amax <= amin:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - amin) / (amax - amin)).astype(np.float32)


def save_combined_map_png(
    signal_arr: np.ndarray,
    background_arr: np.ndarray,
    original_img: np.ndarray,
    out_prefix: Path,
    entry_number: int = None,
    n_pixels: int = None,
    base_signal_score: float = None,
    base_background_score: float = None,
    tag: str = None,
    model_key: str = None,
    method: str = None,
    layer_name: str = None,
    extra_info: str = None,
    cmap: str = "gnuplot_r",
):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(60, 20), dpi=200)

    im0 = ax0.imshow(
        original_img.T,
        origin="lower",
        cmap="jet",
        norm=colors.PowerNorm(gamma=0.35, vmin=original_img.min(), vmax=original_img.max()),
    )
    ax0.set_xlabel("Original Event", fontsize=35, labelpad=20)
    ax0.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
    cbar0 = fig.colorbar(im0, ax=ax0)
    cbar0.set_label("ADC", fontsize=25)
    cbar0.ax.tick_params(labelsize=20)

    im1 = ax1.imshow(signal_arr.T, origin="lower", cmap=cmap, vmin=np.min(signal_arr), vmax=np.max(signal_arr))
    ax1.set_xlabel("Signal Score Map", fontsize=35, labelpad=20)
    ax1.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label("Score", fontsize=25)
    cbar1.ax.tick_params(labelsize=20)

    im2 = ax2.imshow(background_arr.T, origin="lower", cmap=cmap, vmin=np.min(background_arr), vmax=np.max(background_arr))
    ax2.set_xlabel("Background Score Map", fontsize=35, labelpad=20)
    ax2.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label("Score", fontsize=25)
    cbar2.ax.tick_params(labelsize=20)

    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.8, wspace=0.1)

    info_lines = []
    if tag is not None:
        info_lines.append(f"Reason: {tag}")
    if entry_number is not None:
        info_lines.append(f"Entry Number: {entry_number}")
    if n_pixels is not None:
        info_lines.append(f"N Pixels: {n_pixels}")
    if base_signal_score is not None:
        info_lines.append(f"Base Signal Score: {base_signal_score:.4f}")
    if base_background_score is not None:
        info_lines.append(f"Base Background Score: {base_background_score:.4f}")
    if model_key is not None:
        info_lines.append(f"Model: {model_key}")
    if method is not None:
        info_lines.append(f"Method: {method}")
    if layer_name is not None:
        info_lines.append(f"Layer: {layer_name}")

    if info_lines:
        info_text = "\n".join(info_lines)
        fig.text(
            0.5,
            0.01,
            info_text,
            ha="center",
            va="bottom",
            fontsize=18,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.savefig(out_prefix.with_name(out_prefix.name + "_map.png"), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def outputs_exist(out_prefix: Path) -> bool:
    map_png = out_prefix.with_name(out_prefix.name + "_map.png")
    meta_js = out_prefix.with_name(out_prefix.name + "_meta.txt")
    return map_png.exists() and meta_js.exists()


def load_image_from_root(root_file: str, entry: int, plane: int, device: torch.device) -> torch.Tensor:
    from mpid_data import mpid_data_binary
    ds = mpid_data_binary.MPID_Dataset(
        root_file,
        "image2d_image2d_binary_tree",
        device.type,
        plane=plane,
    )
    x = ds[entry][0].view(1, 1, 512, 512)
    return x


# -----------------------------
# Score-CAM
# -----------------------------
class _ForwardActivations:
    def __init__(self, layer: nn.Module):
        self.activations = None
        self.h = layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.activations = out.detach()

    def close(self):
        try:
            self.h.remove()
        except Exception:
            pass


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    name = str(name).strip()
    for n, m in model.named_modules():
        if n == name:
            return m
    raise KeyError(f"Layer '{name}' not found in model.named_modules().")

def pick_default_layer_name(model: nn.Module) -> str:
    names = [n for n, _ in model.named_modules()]
    # ResNet wrapper typically has "net.layer4"
    if "net.layer4" in names:
        return "net.layer4"
    # Some models expose "features"
    if "features" in names:
        return "features.30"

@torch.no_grad()
def scorecam_maps(
    model: nn.Module,
    x: torch.Tensor,              # [1,1,H,W] on device
    layer: nn.Module,
    max_channels: int = 64,
    batch_size: int = 16,
    channel_mode: str = "topk",
    use_relu_acts: bool = True,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: (signal_map, background_map) as numpy float32 arrays at input resolution.
    Weights are sigmoid probs for class 0/1 on masked inputs.
    """
    model.eval()

    H, W = x.shape[-2], x.shape[-1]

    # 1) capture activations on the original input
    hook = _ForwardActivations(layer)
    try:
        _ = model(x)
        acts = hook.activations  # [1,C,h,w]
        if acts is None:
            raise RuntimeError("Failed to capture activations (bad layer name?).")
    finally:
        hook.close()

    acts = acts[0]  # [C,h,w]
    if use_relu_acts:
        acts = torch.relu(acts)

    C, h, w = acts.shape
    if C == 0:
        raise RuntimeError("Activation tensor has zero channels?")

    # 2) choose channels
    energy = torch.mean(torch.abs(acts), dim=(1, 2))  # [C]
    if channel_mode == "topk":
        k = min(int(max_channels), int(C))
        idx = torch.topk(energy, k=k, largest=True).indices
    elif channel_mode == "first":
        k = min(int(max_channels), int(C))
        idx = torch.arange(k, device=acts.device)
    else:
        raise ValueError("--scorecam-channel-mode must be 'topk' or 'first'.")

    # 3) accumulate CAMs at layer resolution (h,w)
    cam_sig = torch.zeros((h, w), device=acts.device, dtype=acts.dtype)
    cam_bkg = torch.zeros((h, w), device=acts.device, dtype=acts.dtype)

    # score in batches
    for s in range(0, idx.numel(), int(batch_size)):
        ii = idx[s : s + int(batch_size)]
        a = acts[ii]  # [B,h,w]
        B = a.shape[0]

        # upsample to input resolution and normalize each mask to [0,1]
        masks = F.interpolate(a.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)  # [B,1,H,W]
        m_min = masks.amin(dim=(2, 3), keepdim=True)
        masks = masks - m_min
        m_max = masks.amax(dim=(2, 3), keepdim=True)
        masks = masks / (m_max + eps)

        xb = x.repeat(B, 1, 1, 1) * masks  # [B,1,H,W]
        logits = model(xb)                 # [B,2]
        probs = torch.sigmoid(logits)      # [B,2]

        w_sig = probs[:, 0].view(B, 1, 1)  # [B,1,1]
        w_bkg = probs[:, 1].view(B, 1, 1)

        cam_sig += torch.sum(w_sig * a, dim=0)
        cam_bkg += torch.sum(w_bkg * a, dim=0)

    # ensure non-negative
    cam_sig = torch.relu(cam_sig)
    cam_bkg = torch.relu(cam_bkg)

    # 4) upsample to input resolution
    cam_sig_up = F.interpolate(cam_sig.view(1, 1, h, w), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    cam_bkg_up = F.interpolate(cam_bkg.view(1, 1, h, w), size=(H, W), mode="bilinear", align_corners=False)[0, 0]

    sig_np = cam_sig_up.detach().cpu().numpy().astype(np.float32)
    bkg_np = cam_bkg_up.detach().cpu().numpy().astype(np.float32)
    return sig_np, bkg_np


# -----------------------------
# CLI / run
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="auto", help="auto | mpid | resnet18_bn | resnet18_gn | resnet34_bn | resnet34_gn")
    p.add_argument("--weight-file", required=True)

    p.add_argument("--input-file", default=None, help="Path to ROOT file (inside container: /data/...)")
    p.add_argument("--entry", type=int, default=None, help="Single entry index in ROOT tree")
    p.add_argument("--entries", type=int, default=1, help="How many entries to run starting at --entry")

    p.add_argument("--from-csv", default=None, help="CSV with columns: root_file, entry_number, out_dir, tag, n_pixels (plus optional weight_file, model, layer_name)")
    p.add_argument("--larcv-base", default="/data", help="Base path inside container for ROOTs (default /data)")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--tag", default="")

    p.add_argument("--adc-lo", type=float, default=10.0)
    p.add_argument("--adc-hi", type=float, default=500.0)

    # NOTE: in this occlusion-style workflow, --normalize means normalize the output maps to [0,1]
    p.add_argument("--normalize", action="store_true")

    p.add_argument("--plane", type=int, default=0)
    p.add_argument("--save-npy", action="store_true")
    p.add_argument("--gpuid", default="0")
    p.add_argument("--n-pixels", type=int, default=None)

    p.add_argument("--layer-name", default=None, help="Layer to hook (e.g. net.layer4.1.conv2). If omitted, choose a reasonable default.")
    p.add_argument("--scorecam-max-channels", type=int, default=64, help="Use up to this many channels from the hooked layer (runtime control).")
    p.add_argument("--scorecam-batch", type=int, default=16, help="How many masked inputs to score per forward batch.")
    p.add_argument("--scorecam-channel-mode", default="topk", choices=["topk", "first"], help="Which channels to use.")
    p.add_argument("--scorecam-no-relu-acts", action="store_true", help="If set, do NOT ReLU the layer activations before using them as masks.")
    return p


def run_one(
    model: nn.Module,
    model_key: str,
    weight_file: str,
    root_file: str,
    entry: int,
    n_pixels: Optional[int],
    out_dir: str,
    tag: str,
    adc_lo: float,
    adc_hi: float,
    normalize: bool,
    plane: int,
    save_npy: bool,
    layer_name: Optional[str],
    scorecam_max_channels: int,
    scorecam_batch: int,
    scorecam_channel_mode: str,
    scorecam_use_relu_acts: bool,
    device: torch.device,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root_name = Path(root_file).name
    if "dirt" in root_name:
        root_name = "dirt"
    elif "nu_overlay" in root_name:
        root_name = "nu_overlay"
    elif "offbeam" in root_name:
        root_name = "offbeam"
    else:
        root_name = ""

    if root_name:
        out_prefix = out_dir / f"scorecam__ENTRY_{entry}__{root_name}"
    else:
        out_prefix = out_dir / f"scorecam__ENTRY_{entry}"

    if outputs_exist(out_prefix):
        print(f"[skip] exists: {out_prefix}")
        return

    # load + clamp like occlusion
    x = load_image_from_root(root_file, entry, plane, device)  # [1,1,512,512] (likely on CPU)
    x = x.to(device)
    x = clamp_adc(x, adc_lo, adc_hi)

    # base scores from unmasked input
    with torch.no_grad():
        logits0 = model(x)
        probs0 = torch.sigmoid(logits0).detach().cpu().numpy()[0]
    base_sig = float(probs0[0])
    base_bkg = float(probs0[1])

    # layer selection
    if layer_name is None:
        layer_name = pick_default_layer_name(model)
        if layer_name == "":
            raise RuntimeError("Could not infer a default layer; please pass --layer-name.")
    layer = get_module_by_name(model, layer_name)

    sig_map, bkg_map = scorecam_maps(
        model=model,
        x=x,
        layer=layer,
        max_channels=int(scorecam_max_channels),
        batch_size=int(scorecam_batch),
        channel_mode=str(scorecam_channel_mode),
        use_relu_acts=bool(scorecam_use_relu_acts),
    )

    if normalize:
        sig_map = norm01(sig_map)
        bkg_map = norm01(bkg_map)

    # prepare original image for plotting (2D)
    original_img = x.detach().cpu().squeeze().numpy()

    extra = f"Channels used: <= {int(scorecam_max_channels)}; batch={int(scorecam_batch)}; mode={scorecam_channel_mode}"
    save_combined_map_png(
        signal_arr=sig_map,
        background_arr=bkg_map,
        original_img=original_img,
        out_prefix=out_prefix,
        entry_number=int(entry),
        n_pixels=(None if n_pixels is None else int(n_pixels)),
        base_signal_score=base_sig,
        base_background_score=base_bkg,
        tag=tag,
        model_key=model_key,
        method="scorecam",
        layer_name=layer_name,
        extra_info=extra,
    )

    meta = {
        "method": "scorecam",
        "model": model_key,
        "weight_file": str(weight_file),
        "input_file": str(root_file),
        "entry_number": int(entry),
        "n_pixels": (None if n_pixels is None else int(n_pixels)),
        "base_signal_score": float(base_sig),
        "base_background_score": float(base_bkg),
        "layer_name": str(layer_name),
        "scorecam_max_channels": int(scorecam_max_channels),
        "scorecam_batch": int(scorecam_batch),
        "scorecam_channel_mode": str(scorecam_channel_mode),
        "scorecam_use_relu_acts": bool(scorecam_use_relu_acts),
        "adc_lo": float(adc_lo),
        "adc_hi": float(adc_hi),
        "normalize": bool(normalize),
        "plane": int(plane),
        "reason": str(tag),
    }
    meta_path = out_prefix.with_name(out_prefix.name + "_meta.txt")
    meta_path.write_text(json.dumps(meta, indent=2))
    print("[ok]", meta_path)

    if save_npy:
        np.save(out_prefix.with_name(out_prefix.name + "_Signal_map.npy"), sig_map)
        np.save(out_prefix.with_name(out_prefix.name + "_Background_map.npy"), bkg_map)


def main():
    args = build_argparser().parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, resolved_key = build_model(args.model, args.weight_file, device)
    print(f"[info] requested_model={args.model} resolved_model={resolved_key} device={device}")

    if args.from_csv:
        import pandas as pd

        df = pd.read_csv(args.from_csv)
        required = {"root_file", "entry_number", "out_dir", "tag"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"--from-csv missing columns: {sorted(missing)}")

        for _, row in df.iterrows():
            root_file = str(row["root_file"])
            if not os.path.isabs(root_file):
                root_file = str(Path(args.larcv_base) / root_file)

            entry = int(row["entry_number"])
            out_dir = str(row["out_dir"])
            tag = str(row["tag"])

            n_pixels = None
            if "n_pixels" in df.columns and not pd.isna(row["n_pixels"]):
                n_pixels = int(row["n_pixels"])

            wfile = (
                str(row["weight_file"])
                if "weight_file" in df.columns and isinstance(row["weight_file"], str)
                else args.weight_file
            )
            mkey = (
                str(row["model"])
                if "model" in df.columns and isinstance(row["model"], str)
                else args.model
            )
            layer_name = (
                str(row["layer_name"])
                if "layer_name" in df.columns and isinstance(row["layer_name"], str)
                else args.layer_name
            )

            if (wfile != args.weight_file) or (mkey != args.model):
                model_i, resolved_i = build_model(mkey, wfile, device)
            else:
                model_i, resolved_i = model, resolved_key

            run_one(
                model=model_i,
                model_key=resolved_i,
                weight_file=wfile,
                root_file=root_file,
                entry=entry,
                n_pixels=n_pixels,
                out_dir=out_dir,
                tag=tag,
                adc_lo=args.adc_lo,
                adc_hi=args.adc_hi,
                normalize=args.normalize,
                plane=args.plane,
                save_npy=args.save_npy,
                layer_name=layer_name,
                scorecam_max_channels=args.scorecam_max_channels,
                scorecam_batch=args.scorecam_batch,
                scorecam_channel_mode=args.scorecam_channel_mode,
                scorecam_use_relu_acts=(not args.scorecam_no_relu_acts),
                device=device,
            )
        return

    # single-entry mode
    if args.input_file is None or args.entry is None:
        raise SystemExit("Need --input-file and --entry unless using --from-csv")

    for e in range(args.entry, args.entry + int(args.entries)):
        run_one(
            model=model,
            model_key=resolved_key,
            weight_file=args.weight_file,
            root_file=args.input_file,
            entry=int(e),
            n_pixels=args.n_pixels,
            out_dir=args.output_dir,
            tag=args.tag or f"{resolved_key}",
            adc_lo=args.adc_lo,
            adc_hi=args.adc_hi,
            normalize=args.normalize,
            plane=args.plane,
            save_npy=args.save_npy,
            layer_name=args.layer_name,
            scorecam_max_channels=args.scorecam_max_channels,
            scorecam_batch=args.scorecam_batch,
            scorecam_channel_mode=args.scorecam_channel_mode,
            scorecam_use_relu_acts=(not args.scorecam_no_relu_acts),
            device=device,
        )


if __name__ == "__main__":
    main()
