#!/usr/bin/env python3
"""
Shared utilities for your XAI CLI scripts (Grad-CAM, Integrated Gradients, Score-CAM).

Goals:
- Single implementation of model loading, IO/plotting, and diagnostics.
- Consistent deletion/insertion curves + overlap diagnostics across methods.
- Keep method-specific scripts thin and easy to maintain.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# -----------------------------
# Diagnostics: general helpers
# -----------------------------

def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def tensor_fingerprint(x: torch.Tensor) -> Dict[str, object]:
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


def gn_summary(model: nn.Module) -> Dict[str, object]:
    """For spotting GN16 vs GN32 mismatches."""
    groups = []
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
            groups.append(int(m.num_groups))
    return {"n_groupnorm": len(groups), "unique_num_groups": sorted(set(groups))}


def score_summary_from_logits(logits: torch.Tensor) -> Dict[str, object]:
    """For BCEWithLogitsLoss-style binary outputs: show logits + sigmoid-per-logit."""
    v = logits.detach().cpu().flatten()
    return {"logits": v.tolist(), "sigmoid_per_logit": torch.sigmoid(v).tolist()}


# -----------------------------
# Helpers: checkpoint + model detection
# -----------------------------

def _extract_state(ckpt) -> Dict[str, torch.Tensor]:
    """Support: raw state_dict, {'state_dict': ...}, {'model_state_dict': ...}, etc."""
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        return ckpt
    raise ValueError("Unsupported checkpoint format (expected dict / state_dict).")


def _looks_like_resnet_state(state: Dict[str, torch.Tensor]) -> bool:
    keys = list(state.keys())
    return any(k.startswith("net.layer") or ".layer" in k for k in keys) or any(
        k.startswith("layer") for k in keys
    )


def _infer_resnet_depth(state: Dict[str, torch.Tensor]) -> int:
    keys = list(state.keys())
    # Heuristic: resnet34 has layer1.2 and layer2.2 blocks, resnet18 doesn't.
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


def _replace_bn_with_gn(module: nn.Module, num_groups: int = 32) -> None:
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
    def __init__(self, depth: int = 34, norm: str = "gn", dropout: float = 0.5):
        super().__init__()
        from torchvision.models import resnet18, resnet34

        net = resnet34(weights=None) if depth == 34 else resnet18(weights=None)
        net.conv1 = nn.Conv2d(
            1, net.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        if norm == "gn":
            _replace_bn_with_gn(net)

        in_features = net.fc.in_features
        net.fc = nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(in_features, 2))
        self.net = net

    def forward(self, x):
        return self.net(x)


def _try_load(
    model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device
) -> Optional[str]:
    try:
        model.to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return None
    except Exception as e:
        return str(e)


def build_model(
    model_key: str, weight_file: str, device: torch.device, dropout: float = 0.5
) -> Tuple[nn.Module, str]:
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
            return ResNetBinaryWrapper(depth=depth, norm=norm, dropout=dropout)
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
# Shared: IO + plotting
# -----------------------------

def clamp_adc(img: torch.Tensor, adc_lo: float, adc_hi: float) -> torch.Tensor:
    img = img.clone()
    img[img > adc_hi] = adc_hi
    img[img < adc_lo] = 0.0
    return img


def rootname_simplify(root_file: str) -> str:
    root_name = Path(root_file).name
    if "dirt" in root_name:
        return "dirt"
    if "nu_overlay" in root_name:
        return "nu_overlay"
    if "offbeam" in root_name:
        return "offbeam"
    return ""


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
    entry_number: int | None = None,
    n_pixels: int | None = None,
    base_signal_score: float | None = None,
    base_background_score: float | None = None,
    tag: str | None = None,
    model_key: str | None = None,
    method: str | None = None,
    layer_name: str | None = None,
    extra_info: str | None = None,
    cmap: str = "gnuplot_r",
    signed_cmap: str = "seismic",
    vis_percentile: float = 99.5,
    use_original_nonzero_mask: bool = True,
) -> None:
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(60, 20), dpi=200)

    im0 = ax0.imshow(
        original_img.T,
        origin="lower",
        cmap="jet",
        norm=colors.PowerNorm(
            gamma=0.35, vmin=original_img.min(), vmax=original_img.max()
        ),
    )
    ax0.set_xlabel("Original Event", fontsize=35, labelpad=20)
    ax0.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
    cbar0 = fig.colorbar(im0, ax=ax0)
    cbar0.set_label("ADC", fontsize=25)
    cbar0.ax.tick_params(labelsize=20)

    # --- attribution map visualization ---
    # For sparse detector images, min-max scaling often makes everything look flat.
    # Use percentile scaling, optionally restricted to the nonzero region of the original image.
    def _masked_vals(arr: np.ndarray) -> np.ndarray:
        if not use_original_nonzero_mask:
            return arr.reshape(-1)
        m = (original_img != 0)
        v = arr[m]
        return v.reshape(-1) if v.size else arr.reshape(-1)

    def _is_signed(arr: np.ndarray) -> bool:
        return float(np.min(arr)) < 0.0 and float(np.max(arr)) > 0.0

    def _norm_for(arr: np.ndarray):
        v = _masked_vals(arr)
        if _is_signed(arr):
            vmax = float(np.percentile(np.abs(v), vis_percentile)) if v.size else float(np.max(np.abs(arr)))
            vmax = max(vmax, 1e-12)
            return colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), signed_cmap
        else:
            vmin = float(np.percentile(v, 100.0 - vis_percentile)) if v.size else float(np.min(arr))
            vmax = float(np.percentile(v, vis_percentile)) if v.size else float(np.max(arr))
            if vmax <= vmin:
                vmax = vmin + 1e-12
            return colors.Normalize(vmin=vmin, vmax=vmax), cmap

    norm1, cmap1 = _norm_for(signal_arr)
    im1 = ax1.imshow(signal_arr.T, origin="lower", cmap=cmap1, norm=norm1)
    ax1.set_xlabel("Signal Score Map", fontsize=35, labelpad=20)
    ax1.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label("Score", fontsize=25)
    cbar1.ax.tick_params(labelsize=20)

    norm2, cmap2 = _norm_for(background_arr)
    im2 = ax2.imshow(background_arr.T, origin="lower", cmap=cmap2, norm=norm2)
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
    if extra_info:
        info_lines.append(str(extra_info))

    if info_lines:
        fig.text(
            0.5,
            0.01,
            "\n".join(info_lines),
            ha="center",
            va="bottom",
            fontsize=18,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    suffix = f"_map_{layer_name}.png" if layer_name else "_map.png"
    fig.savefig(out_prefix.with_name(out_prefix.name + suffix), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def outputs_exist(out_prefix: Path, layer_name: str | None = None) -> bool:
    map_png = out_prefix.with_name(out_prefix.name + (f"_map_{layer_name}.png" if layer_name else "_map.png"))
    meta_js = out_prefix.with_name(out_prefix.name + (f"_meta_{layer_name}.txt" if layer_name else "_meta.txt"))
    return map_png.exists() and meta_js.exists()


def load_image_from_root(
    root_file: str, entry: int, plane: int, device: torch.device, diag_print: bool = True
) -> torch.Tensor:
    from mpid_data import mpid_data_binary
    ds = mpid_data_binary.MPID_Dataset(
        root_file, "image2d_image2d_binary_tree", device.type, plane=plane
    )
    x, y, info, nevents = ds[entry]

    if diag_print:
        print("\n--- DIAG event ---")
        print("ENTRY:", entry)
        print("event_info [run,subrun,event]:", info)
        print("nevents:", nevents)
        print("label y:", y.tolist() if hasattr(y, "tolist") else y)
        print("------------------\n")

    x = x.view(1, 1, 512, 512)
    if diag_print:
        print("x_raw fingerprint:", tensor_fingerprint(x))
    return x


# -----------------------------
# CAM layer selection (shared)
# -----------------------------

def preset_layer_name(model: nn.Module, preset: str = "mid32") -> str:
    """
    Return a sensible layer name for CAM hooks, matched by *spatial grid size* across MPID vs ResNet.

      - mid128: ResNet layer1, MPID features.7
      - mid64:  ResNet layer2, MPID features.14
      - mid32:  ResNet layer3, MPID features.21
      - final:  ResNet layer4, MPID features.31 (final conv stage)

    You can always override with --layer-name.
    """
    preset = (preset or "mid32").lower()
    names = set(n for n, _ in model.named_modules())
    is_resnet = any(n.startswith("net.layer") for n in names)

    if is_resnet:
        mapping = {"mid128": "net.layer1", "mid64": "net.layer2", "mid32": "net.layer3", "final": "net.layer4"}
        cand = mapping.get(preset, "net.layer3")
        return cand if cand in names else "net.layer4"

    mapping = {"mid128": "features.7", "mid64": "features.14", "mid32": "features.21", "final": "features.31"}
    cand = mapping.get(preset, "features.21")
    if cand in names:
        return cand

    # fallback: last conv2d
    last_conv = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = n
    return last_conv or "features.21"


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    name = str(name).strip()
    for n, m in model.named_modules():
        if n == name:
            return m
    raise KeyError(
        f"Layer '{name}' not found. Examples: net.layer4, net.layer4.1.conv2, features.0"
    )


# -----------------------------
# Diagnostics: overlap + curves (single source of truth)
# -----------------------------

def active_mask_from_x(x: torch.Tensor, threshold: float = 0.0) -> np.ndarray:
    """x: [1,1,H,W] -> active mask on CPU based on clamped ADC."""
    arr = x.detach().cpu().numpy()[0, 0]
    return arr > float(threshold)


def topk_mask(attr: np.ndarray, frac: float, use_abs: bool = True) -> np.ndarray:
    """Boolean mask for top frac pixels by |attr| (default) or attr."""
    a = np.abs(attr) if use_abs else attr
    flat = a.reshape(-1)
    n = flat.size
    if n == 0:
        return np.zeros_like(a, dtype=bool)

    frac = float(frac)
    frac = min(max(frac, 0.0), 1.0)
    k = int(round(frac * n))
    if k <= 0:
        return np.zeros_like(a, dtype=bool)
    if k >= n:
        return np.ones_like(a, dtype=bool)

    idx = np.argpartition(flat, -k)[-k:]
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m.reshape(a.shape)


def overlap_topk_on_active(
    attr: np.ndarray, active: np.ndarray, topk_frac: float, use_abs: bool = True
) -> float:
    topk = topk_mask(attr, topk_frac, use_abs=use_abs)
    denom = float(topk.sum())
    if denom <= 0:
        return float("nan")
    return float((topk & active).sum()) / denom


def auc_trapezoid(y, x) -> float:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.size < 2:
        return float("nan")
    return float(np.trapezoid(y, x))


def deletion_insertion_curves(
    model: nn.Module,
    x: torch.Tensor,
    attr_rank: np.ndarray,
    signal_idx: int = 0,
    background_idx: int = 1,
    n_steps: int = 11,
    seed: int = 123,
    baseline: Optional[torch.Tensor] = None,
    use_abs_rank: bool = True,
) -> Dict[str, object]:
    """Faithfulness curves using sigmoid(logit[idx]) for each class.

    - Deletion: progressively set top-k% ranked pixels -> baseline.
    - Insertion: start at baseline and insert top-k% ranked pixels.
    - Also computes random baseline curves using the same fractions.
    - Ranking pixels uses |attr_rank| by default (use_abs_rank=True).

    Returns a stable schema used across all scripts.
    """
    if baseline is None:
        baseline = torch.zeros_like(x)

    H, W = attr_rank.shape
    a = np.abs(attr_rank) if use_abs_rank else attr_rank
    flat = a.reshape(-1)
    n = flat.size
    order = np.argsort(flat)[::-1]

    rng = np.random.default_rng(int(seed))
    rand_order = rng.permutation(n)

    fracs = np.linspace(0.0, 1.0, int(n_steps))
    out = {
        "fractions": fracs.tolist(),
        "deletion": {"signal": [], "background": []},
        "insertion": {"signal": [], "background": []},
        "random_deletion": {"signal": [], "background": []},
        "random_insertion": {"signal": [], "background": []},
    }

    model.eval()

    # re-use flat masks to reduce allocations
    mask_flat = np.zeros(n, dtype=bool)
    rmask_flat = np.zeros(n, dtype=bool)

    with torch.no_grad():
        for frac in fracs:
            k = int(round(float(frac) * n))
            mask_flat[:] = False
            rmask_flat[:] = False
            if k > 0:
                mask_flat[order[:k]] = True
                rmask_flat[rand_order[:k]] = True

            mask = torch.from_numpy(mask_flat.reshape(H, W)).to(x.device)
            rmask = torch.from_numpy(rmask_flat.reshape(H, W)).to(x.device)
            mask4 = mask.unsqueeze(0).unsqueeze(0)
            rmask4 = rmask.unsqueeze(0).unsqueeze(0)

            x_del = x.clone()
            x_del[mask4] = baseline[mask4]
            x_ins = baseline + (x - baseline) * mask4.float()

            xr_del = x.clone()
            xr_del[rmask4] = baseline[rmask4]
            xr_ins = baseline + (x - baseline) * rmask4.float()

            p_del = torch.sigmoid(model(x_del))[0]
            p_ins = torch.sigmoid(model(x_ins))[0]
            pr_del = torch.sigmoid(model(xr_del))[0]
            pr_ins = torch.sigmoid(model(xr_ins))[0]

            out["deletion"]["signal"].append(float(p_del[signal_idx]))
            out["deletion"]["background"].append(float(p_del[background_idx]))
            out["insertion"]["signal"].append(float(p_ins[signal_idx]))
            out["insertion"]["background"].append(float(p_ins[background_idx]))

            out["random_deletion"]["signal"].append(float(pr_del[signal_idx]))
            out["random_deletion"]["background"].append(float(pr_del[background_idx]))
            out["random_insertion"]["signal"].append(float(pr_ins[signal_idx]))
            out["random_insertion"]["background"].append(float(pr_ins[background_idx]))

    # AUC summaries
    out["auc"] = {
        "deletion_signal": auc_trapezoid(out["deletion"]["signal"], fracs),
        "deletion_background": auc_trapezoid(out["deletion"]["background"], fracs),
        "insertion_signal": auc_trapezoid(out["insertion"]["signal"], fracs),
        "insertion_background": auc_trapezoid(out["insertion"]["background"], fracs),
        "random_deletion_signal": auc_trapezoid(out["random_deletion"]["signal"], fracs),
        "random_deletion_background": auc_trapezoid(out["random_deletion"]["background"], fracs),
        "random_insertion_signal": auc_trapezoid(out["random_insertion"]["signal"], fracs),
        "random_insertion_background": auc_trapezoid(out["random_insertion"]["background"], fracs),
    }
    return out
