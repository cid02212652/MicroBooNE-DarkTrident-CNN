#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

# Matplotlib only for saving plots (Agg backend safe on batch)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# -----------------------------
# Diagnostics: functions to help
# -----------------------------

import os, sys, hashlib
import numpy as np
import torch
import torch.nn as nn

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def tensor_fingerprint(x: torch.Tensor):
    """Stable-ish fingerprint for 'is this the exact same tensor?' checks."""
    x = x.detach().cpu().contiguous()
    arr = x.numpy()
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "sum": float(arr.sum()),
        "nnz": int((arr != 0).sum()),
        "sha1": hashlib.sha1(arr.tobytes()).hexdigest(),
    }

def gn_summary(model: nn.Module):
    """For spotting GN16 vs GN32 mismatches."""
    groups = []
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
            groups.append(int(m.num_groups))
    return {
        "n_groupnorm": len(groups),
        "unique_num_groups": sorted(set(groups)),
    }

def score_summary_from_logits(logits: torch.Tensor):
    """
    Prints multiple common "binary score" conventions so we can spot a mismatch:
      A) sigmoid per-logit (your current approach)
      B) softmax over 2 logits
      C) sigmoid(logit0 - logit1)
    """
    v = logits.detach().cpu().flatten()
    out = {"logits": v.tolist()}

    # A: sigmoid per-logit
    sig = torch.sigmoid(v)
    out["sigmoid_per_logit"] = sig.tolist()

    # B: softmax over 2
    if v.numel() == 2:
        sm = torch.softmax(v, dim=0)
        out["softmax_2class"] = sm.tolist()
        out["sigmoid_logitdiff_(l0-l1)"] = float(torch.sigmoid(v[0] - v[1]))
    return out

# -----------------------------
# Helpers: checkpoint + model detection
# -----------------------------

def _extract_state(ckpt):
    # support: raw state_dict, {"state_dict": ...}, {"model_state_dict": ...}
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    if isinstance(ckpt, dict):
        # might already be a state_dict
        return ckpt
    raise ValueError("Unsupported checkpoint format (expected dict / state_dict).")

def _looks_like_resnet_state(state: Dict[str, torch.Tensor]) -> bool:
    keys = list(state.keys())
    return any(k.startswith("net.layer") or ".layer" in k for k in keys) or any(k.startswith("layer") for k in keys)

def _infer_resnet_depth(state: Dict[str, torch.Tensor]) -> int:
    # ResNet34 has blocks like layer1.2, layer2.3, layer3.5, etc.
    keys = list(state.keys())
    for k in keys:
        # common: net.layer1.2.conv1.weight
        if ".layer1.2." in k or "layer1.2." in k:
            return 34
        if ".layer2.2." in k or "layer2.2." in k:
            # could be 34 or bigger, but for your case it's 34
            return 34
    return 18

def _infer_norm(state: Dict[str, torch.Tensor]) -> str:
    # BN has running_mean/running_var buffers in the state_dict
    keys = list(state.keys())
    if any(k.endswith("running_mean") or k.endswith("running_var") for k in keys):
        return "bn"
    return "gn"

def _summarise_ckpt(state: Dict[str, torch.Tensor]) -> str:
    is_resnet = _looks_like_resnet_state(state)
    if not is_resnet:
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
    """
    Recursively replace all BatchNorm2d with GroupNorm.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features, eps=child.eps, affine=True)
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

        # your images are 1-channel (shape [N,1,512,512])
        net.conv1 = nn.Conv2d(1, net.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

        if norm == "gn":
            _replace_bn_with_gn(net)

        # 2 logits (signal/background) — we output logits and sigmoid later
        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 2)
        )

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
      - tries the requested model_key first
      - if that fails, auto-detects from checkpoint and retries
      - if still fails, tries all 4 resnet variants (18/34 x bn/gn) if it's resnet-like
    Returns: (model, resolved_model_key)
    """
    model_key = (model_key or "auto").lower()
    ckpt = torch.load(weight_file, map_location=device)
    state = _extract_state(ckpt)

    # Decide what checkpoint "looks like"
    guessed = _summarise_ckpt(state)

    def make(key: str) -> nn.Module:
        if key in ["mpid", "dmcnn", "dm-cnn"]:
            return MPIDBinary().net
        if key.startswith("resnet"):
            # key like resnet18_bn
            depth = 34 if "34" in key else 18
            norm = "gn" if key.endswith("_gn") else "bn"
            return ResNetBinaryWrapper(depth=depth, norm=norm, dropout=0.0)
        raise ValueError(f"Unknown model key: {key}")

    # 1) try requested
    if model_key != "auto":
        model = make(model_key)
        err = _try_load(model, state, device)
        if err is None:
            return model, model_key

    # 2) try guessed
    model = make(guessed)
    err = _try_load(model, state, device)
    if err is None:
        return model, guessed

    # 3) brute-force for resnet-like ckpts
    if _looks_like_resnet_state(state):
        candidates = ["resnet18_bn", "resnet18_gn", "resnet34_bn", "resnet34_gn"]
        for c in candidates:
            model = make(c)
            err2 = _try_load(model, state, device)
            if err2 is None:
                return model, c

    # 4) if none worked, raise the original error with context
    raise RuntimeError(
        f"Failed to load weights.\n"
        f"Requested={model_key}, Guessed={guessed}\n"
        f"Last error:\n{err}"
    )

# -----------------------------
# Occlusion logic
# -----------------------------

def clamp_adc(img: torch.Tensor, adc_lo: float, adc_hi: float) -> torch.Tensor:
    img = img.clone()
    img[img > adc_hi] = adc_hi
    img[img < adc_lo] = 0.0
    return img

def occlude_scan(
    model: nn.Module,
    input_image: torch.Tensor,   # shape [1,1,H,W]
    occlusion_size: int = 4,
    stride: int = 1,
    adc_lo: float = 10.0,
    adc_hi: float = 500.0,
    normalize: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, np.ndarray]:
    """
    Returns:
      signal_map, background_map (H, W) float arrays (optionally normalized 0..1)
      base_scores dict
    """
    H, W = input_image.shape[-2], input_image.shape[-1]
    pad = occlusion_size

    x0 = input_image.to(device)
    x0 = clamp_adc(x0, adc_lo, adc_hi)
    print("x_clamped fingerprint:", tensor_fingerprint(x0))

    with torch.no_grad():
        logits0 = model(x0)
        probs0 = torch.sigmoid(logits0).detach().cpu().numpy()[0]
        
    print("\n--- DIAG forward ---")
    print(score_summary_from_logits(logits0))
    print("--------------------\n")
    
    base_sig = float(probs0[0])
    base_bkg = float(probs0[1])

    sig_map = np.full((H, W), base_sig, dtype=np.float32)
    bkg_map = np.full((H, W), base_bkg, dtype=np.float32)

    # scan
    for x in range(pad, H - pad, stride):
        for y in range(pad, W - pad, stride):
            if float(x0[0,0,x,y].detach().cpu()) == 0.0:
                continue

            x_occ = x0.detach().cpu().clone()
            x_occ[0,0, x-pad:x+pad+1, y-pad:y+pad+1] = 0.0

            with torch.no_grad():
                logits = model(x_occ.to(device))
                probs = torch.sigmoid(logits).detach().cpu().numpy()[0]

            sig_map[x, y] = float(probs[0])
            bkg_map[x, y] = float(probs[1])

    if normalize:
        def norm01(a):
            amin, amax = float(np.min(a)), float(np.max(a))
            if amax <= amin:
                return np.zeros_like(a)
            return (a - amin) / (amax - amin)
        sig_map = norm01(sig_map).astype(np.float32)
        bkg_map = norm01(bkg_map).astype(np.float32)

    return {
        "signal_map": sig_map,
        "background_map": bkg_map,
        "base_signal_score": base_sig,
        "base_background_score": base_bkg,
    }

# def save_map_png(arr: np.ndarray, out_prefix: Path, which: str, cmap: str = "gnuplot_r"):
#     fig, ax = plt.subplots(figsize=(20, 20), dpi=200)
#     im = ax.imshow(arr.T, origin="lower", cmap=cmap, vmin=np.min(arr), vmax=np.max(arr))
#     ax.set_xlabel(f"{which} Score Map", fontsize=35, labelpad=20)
#     ax.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label("score", fontsize=25)
#     cbar.ax.tick_params(labelsize=20)

#     fig.savefig(out_prefix.with_name(out_prefix.name + f"_{which}_map.png"), bbox_inches="tight")
#     plt.close(fig)

# def save_map_png(arr: np.ndarray, out_prefix: Path, which: str, 
#                  entry_number: int = None, n_pixels: int = None, 
#                  base_signal_score: float = None, base_background_score: float = None,
#                  tag: str = None, cmap: str = "gnuplot_r"):
#     fig, ax = plt.subplots(figsize=(20, 20), dpi=200)
#     im = ax.imshow(arr.T, origin="lower", cmap=cmap, vmin=np.min(arr), vmax=np.max(arr))
#     ax.set_xlabel(f"{which} Score Map", fontsize=35, labelpad=20)
#     ax.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label("score", fontsize=25)
#     cbar.ax.tick_params(labelsize=20)

#     # Add metadata text below the plot (vertically stacked)
#     info_lines = []
#     if tag is not None:
#         info_lines.append(f"Reason: {tag}")
#     if entry_number is not None:
#         info_lines.append(f"Entry Number: {entry_number}")
#     if n_pixels is not None:
#         info_lines.append(f"N Pixels: {n_pixels}")
    
#     # Show appropriate base score based on which plot
#     if "Signal" in which and base_signal_score is not None:
#         info_lines.append(f"Base Signal Score: {base_signal_score:.4f}")
#     elif "Background" in which and base_background_score is not None:
#         info_lines.append(f"Base Background Score: {base_background_score:.4f}")
    
#     if info_lines:
#         info_text = "\n".join(info_lines)  # Newlines for vertical stacking
#         fig.text(0.5, 0.02, info_text, ha='center', va='top', fontsize=18, 
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

#     fig.savefig(out_prefix.with_name(out_prefix.name + f"_{which}_map.png"), 
#             bbox_inches="tight", pad_inches=0.3)
#     plt.close(fig)

# def save_combined_map_png(signal_arr: np.ndarray, background_arr: np.ndarray, 
#                           out_prefix: Path,
#                           entry_number: int = None, n_pixels: int = None, 
#                           base_signal_score: float = None, base_background_score: float = None,
#                           tag: str = None, cmap: str = "gnuplot_r"):
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 20), dpi=200)
    
#     # Signal map (left)
#     im1 = ax1.imshow(signal_arr.T, origin="lower", cmap=cmap, 
#                      vmin=np.min(signal_arr), vmax=np.max(signal_arr))
#     ax1.set_xlabel("Signal Score Map", fontsize=35, labelpad=20)
#     ax1.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
#     cbar1 = fig.colorbar(im1, ax=ax1)
#     cbar1.set_label("score", fontsize=25)
#     cbar1.ax.tick_params(labelsize=20)
    
#     # Background map (right)
#     im2 = ax2.imshow(background_arr.T, origin="lower", cmap=cmap, 
#                      vmin=np.min(background_arr), vmax=np.max(background_arr))
#     ax2.set_xlabel("Background Score Map", fontsize=35, labelpad=20)
#     ax2.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
#     cbar2 = fig.colorbar(im2, ax=ax2)
#     cbar2.set_label("score", fontsize=25)
#     cbar2.ax.tick_params(labelsize=20)
    
#     # Add metadata text below the plots
#     info_lines = []
#     if tag is not None:
#         info_lines.append(f"Reason: {tag}")
#     if entry_number is not None:
#         info_lines.append(f"Entry Number: {entry_number}")
#     if n_pixels is not None:
#         info_lines.append(f"N Pixels: {n_pixels}")
#     if base_signal_score is not None:
#         info_lines.append(f"Base Signal Score: {base_signal_score:.4f}")
#     if base_background_score is not None:
#         info_lines.append(f"Base Background Score: {base_background_score:.4f}")
    
#     if info_lines:
#         info_text = "\n".join(info_lines)
#         fig.text(0.5, 0.02, info_text, ha='center', va='top', fontsize=18, 
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     fig.savefig(out_prefix.with_name(out_prefix.name + "_map.png"), 
#                 bbox_inches="tight", pad_inches=0.3)
#     plt.close(fig)
    
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
    model_key=None,
    method=None,
    cmap: str = "gnuplot_r",
):
    # Create 3 subplots: original, signal, background
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(60, 20), dpi=200)
    
    # Original image (left) - using jet colormap and power norm like PrintImage
    im0 = ax0.imshow(original_img.T, origin="lower", cmap='jet', 
                     norm=colors.PowerNorm(gamma=0.35, vmin=original_img.min(), vmax=original_img.max()))
    ax0.set_xlabel("Original Event", fontsize=35, labelpad=20)
    ax0.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
    cbar0 = fig.colorbar(im0, ax=ax0)
    cbar0.set_label("ADC", fontsize=25)
    cbar0.ax.tick_params(labelsize=20)
    
    # Signal occlusion map (middle) - independent scale
    im1 = ax1.imshow(signal_arr.T, origin="lower", cmap=cmap, 
                     vmin=np.min(signal_arr), vmax=np.max(signal_arr))
    ax1.set_xlabel("Signal Score Map", fontsize=35, labelpad=20)
    ax1.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label("Score", fontsize=25)
    cbar1.ax.tick_params(labelsize=20)
    
    # Background occlusion map (right) - independent scale
    im2 = ax2.imshow(background_arr.T, origin="lower", cmap=cmap, 
                     vmin=np.min(background_arr), vmax=np.max(background_arr))
    ax2.set_xlabel("Background Score Map", fontsize=35, labelpad=20)
    ax2.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label("Score", fontsize=25)
    cbar2.ax.tick_params(labelsize=20)
    
    fig.subplots_adjust(
        left=0.15,   # bigger left margin
        right=0.85,  # bigger right margin
        bottom=0.2,  # bigger bottom margin
        top=0.8,     # bigger top margin
        wspace=0.1, # keep or increase gap between panels
    )
    
    # Add metadata text below
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
    
    if info_lines:
        info_text = "\n".join(info_lines)
        fig.text(0.5, 0.01, info_text, ha='center', va='bottom', fontsize=18, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.savefig(out_prefix.with_name(out_prefix.name + "_map.png"), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def outputs_exist(out_prefix: Path) -> bool:
    # sig_png = out_prefix.with_name(out_prefix.name + "_Signal_map.png")
    # bkg_png = out_prefix.with_name(out_prefix.name + "_Background_map.png")
    map_png = out_prefix.with_name(out_prefix.name + "_map.png")
    meta_js = out_prefix.with_name(out_prefix.name + "_meta.txt")
    # require meta + both pngs as "done"
    # return sig_png.exists() and bkg_png.exists() and meta_js.exists()
    return map_png.exists() and meta_js.exists()

# -----------------------------
# CLI
# -----------------------------

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="auto", help="auto | mpid | resnet18_bn | resnet18_gn | resnet34_bn | resnet34_gn")
    p.add_argument("--weight-file", required=True)
    p.add_argument("--input-file", default=None, help="Path to ROOT file (inside container: /data/...)")
    p.add_argument("--entry", type=int, default=None, help="Single entry index in ROOT tree")
    p.add_argument("--entries", type=int, default=1, help="How many entries to run starting at --entry")
    p.add_argument("--from-csv", default=None, help="CSV with columns: root_file, entry_number, out_dir, tag, weight_file(optional), model(optional)")
    p.add_argument("--larcv-base", default="/data", help="Base bind inside container for ROOTs (default /data)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--occlusion-size", type=int, default=4)
    p.add_argument("--adc-lo", type=float, default=10.0)
    p.add_argument("--adc-hi", type=float, default=500.0)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--plane", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--save-npy", action="store_true")
    p.add_argument("--gpuid", default="0")
    p.add_argument("--n-pixels", type=int, default=None)
    return p

# def load_image_from_root(root_file: str, entry: int, plane: int, device: torch.device) -> torch.Tensor:
#     from mpid_data import mpid_data_binary
#     ds = mpid_data_binary.MPID_Dataset(root_file, "image2d_image2d_binary_tree", device.type, plane=plane)
#     x = ds[entry][0].view(1,1,512,512)
#     return x

def load_image_from_root(root_file: str, entry: int, plane: int, device: torch.device) -> torch.Tensor:
    from mpid_data import mpid_data_binary
    ds = mpid_data_binary.MPID_Dataset(root_file, "image2d_image2d_binary_tree", device.type, plane=plane)
    x, y, info, nevents = ds[entry]  # or ds[entry]
    print("\n--- DIAG event ---")
    print("ENTRY:", entry)
    print("event_info [run,subrun,event]:", info)
    print("nevents:", nevents)
    print("label y:", y.tolist() if hasattr(y, "tolist") else y)
    print("------------------\n")
    x = x.view(1, 1, 512, 512)  # match what you actually feed the network
    print("x_raw fingerprint:", tensor_fingerprint(x))
    return x

def run_one(model, model_key, weight_file, root_file, entry, n_pixels, out_dir, tag,
            occlusion_size, stride, adc_lo, adc_hi, normalize, plane, save_npy, device):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root_name = Path(root_file).name

    # Extract simplified sample type
    if "dirt" in root_name:
        root_name = "dirt"
    elif "nu_overlay" in root_name:
        root_name = "nu_overlay"
    elif "offbeam" in root_name:
        root_name = "offbeam"
    else:
        root_name = ""  # For the dt_ratio file

    # Standardised prefix (matches your resnet outputs style)
    # Adjust out_prefix to handle empty root_name
    if root_name:
        out_prefix = out_dir / f"occlusion__ENTRY_{entry}__{root_name}"
    else:
        out_prefix = out_dir / f"occlusion__ENTRY_{entry}"

    if outputs_exist(out_prefix):
        print(f"[skip] exists: {out_prefix}")
        return

    x = load_image_from_root(root_file, entry, plane, device)

    res = occlude_scan(
        model=model,
        input_image=x,
        occlusion_size=occlusion_size,
        stride=stride,
        adc_lo=adc_lo,
        adc_hi=adc_hi,
        normalize=normalize,
        device=device
    )

    # save_map_png(res["signal_map"], out_prefix, "Signal")
    # save_map_png(res["background_map"], out_prefix, "Background")

    # save_map_png(
    #     arr=res["signal_map"], 
    #     out_prefix=out_prefix, 
    #     which="Signal",
    #     entry_number=entry,
    #     n_pixels=n_pixels,
    #     base_signal_score=res["base_signal_score"],
    #     base_background_score=res["base_background_score"],
    #     tag=tag
    # )
    
    # save_map_png(
    #     arr=res["background_map"], 
    #     out_prefix=out_prefix, 
    #     which="Background",
    #     entry_number=entry,
    #     n_pixels=n_pixels,
    #     base_signal_score=res["base_signal_score"],
    #     base_background_score=res["base_background_score"],
    #     tag=tag
    # )

    # save_combined_map_png(
    #     signal_arr=res["signal_map"], 
    #     background_arr=res["background_map"],
    #     out_prefix=out_prefix,
    #     entry_number=entry,
    #     n_pixels=n_pixels,
    #     base_signal_score=res["base_signal_score"],
    #     base_background_score=res["base_background_score"],
    #     tag=tag
    # )

    # Prepare the original image for plotting
    original_img = x.clone()  # Make a copy
    original_img = clamp_adc(original_img, adc_lo, adc_hi)  # Apply same clamping
    original_img = original_img.squeeze().cpu().numpy()  # Convert to 2D numpy [H, W]

    save_combined_map_png(
        signal_arr=res["signal_map"], 
        background_arr=res["background_map"],
        original_img=original_img,  # the preprocessed input image
        out_prefix=out_prefix,
        entry_number=entry,
        n_pixels=n_pixels,
        base_signal_score=res["base_signal_score"],
        base_background_score=res["base_background_score"],
        tag=tag,
        model_key=model_key,
        method="occlusion",
    )

    meta = {
        "model": model_key,
        "weight_file": str(weight_file),
        "input_file": str(root_file),
        "entry_number": int(entry),
        "n_pixels": int(n_pixels),
        "base_signal_score": float(res["base_signal_score"]),
        "base_background_score": float(res["base_background_score"]),
        "method": "occlusion",
        "occlusion_size": int(occlusion_size),
        "stride": int(stride),
        "adc_lo": float(adc_lo),
        "adc_hi": float(adc_hi),
        "normalize": bool(normalize),
        "plane": int(plane),
        "reason": tag,
    }
    
    meta_path = out_prefix.with_name(out_prefix.name + "_meta.txt")
    meta_path.write_text(json.dumps(meta, indent=2))
    print("[ok]", meta_path)

    if save_npy:
        np.save(out_prefix.with_name(out_prefix.name + "_Signal_map.npy"), res["signal_map"])
        np.save(out_prefix.with_name(out_prefix.name + "_Background_map.npy"), res["background_map"])

def main():
    args = build_argparser().parse_args()

    print("\n================ DIAG HEADER ================")
    print("python:", sys.executable)
    print("cwd:", os.getcwd())
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("============================================\n")
    
    print("DIAG args:",
          {"input_file": args.input_file, "entry": args.entry, "plane": args.plane,
           "weight_file": args.weight_file, "model_key": getattr(args, "model_key", None),
           "adc_lo": getattr(args, "adc_lo", None), "adc_hi": getattr(args, "adc_hi", None),
           "normalize": getattr(args, "normalize", None)})
    
    print("\n--- DIAG weights ---")
    print("weight_file:", args.weight_file)
    print("exists:", os.path.exists(args.weight_file))
    if os.path.exists(args.weight_file):
        print("size_bytes:", os.path.getsize(args.weight_file))
        print("sha256:", sha256_file(args.weight_file))
    print("--------------------\n")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model ONCE per invocation (fast)
    model, resolved_key = build_model(args.model, args.weight_file, device)
    print(f"[info] requested_model={args.model} resolved_model={resolved_key}")

    print("\n--- DIAG model ---")
    print("resolved_model_key:", resolved_key)
    print("model_type:", type(model))
    print("gn_summary:", gn_summary(model))
    print("------------------\n")

    if args.from_csv:
        import pandas as pd
        df = pd.read_csv(args.from_csv)

        required = {"root_file", "entry_number", "n_pixels", "out_dir", "tag"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"--from-csv missing columns: {sorted(missing)}")

        for _, row in df.iterrows():
            root_file = str(row["root_file"])
            entry = int(row["entry_number"])
            n_pixels = int(row["n_pixels"])
            out_dir = str(row["out_dir"])
            tag = str(row["tag"])

            # allow per-row override (optional)
            wfile = str(row["weight_file"]) if "weight_file" in df.columns and isinstance(row["weight_file"], str) else args.weight_file
            mkey  = str(row["model"]) if "model" in df.columns and isinstance(row["model"], str) else args.model

            # rebuild only if needed
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
                occlusion_size=args.occlusion_size,
                stride=args.stride,
                adc_lo=args.adc_lo,
                adc_hi=args.adc_hi,
                normalize=args.normalize,
                plane=args.plane,
                save_npy=args.save_npy,
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
            entry=e,
            n_pixels=args.n_pixels,
            out_dir=args.output_dir,
            tag=args.tag or f"{resolved_key}",
            occlusion_size=args.occlusion_size,
            stride=args.stride,
            adc_lo=args.adc_lo,
            adc_hi=args.adc_hi,
            normalize=args.normalize,
            plane=args.plane,
            save_npy=args.save_npy,
            device=device,
        )

if __name__ == "__main__":
    main()
