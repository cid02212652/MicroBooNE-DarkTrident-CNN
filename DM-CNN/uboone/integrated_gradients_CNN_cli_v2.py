#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

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
    """For BCEWithLogitsLoss-style binary outputs: show logits + sigmoid-per-logit."""
    v = logits.detach().cpu().flatten()
    out = {
        "logits": v.tolist(),
        "sigmoid_per_logit": torch.sigmoid(v).tolist(),
    }
    return out

# -----------------------------
# Helpers: checkpoint + model detection (copied to match occlusion style)
# -----------------------------

def _extract_state(ckpt):
    # support: raw state_dict, {"state_dict": ...}, {"model_state_dict": ...}
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        return ckpt
    raise ValueError("Unsupported checkpoint format (expected dict / state_dict).")


def _looks_like_resnet_state(state: Dict[str, torch.Tensor]) -> bool:
    keys = list(state.keys())
    return any(k.startswith("net.layer") or ".layer" in k for k in keys) or any(k.startswith("layer") for k in keys)


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
# Models (copied to match occlusion style)
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
                affine=True
            )
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, num_groups=num_groups)


class ResNetBinaryWrapper(nn.Module):
    def __init__(self, depth: int = 18, norm: str = "bn", dropout: float = 0.0):
        super().__init__()
        from torchvision.models import resnet18, resnet34
        net = resnet34(weights=None) if depth == 34 else resnet18(weights=None)

        net.conv1 = nn.Conv2d(1, net.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        if norm == "gn":
            _replace_bn_with_gn(net)

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

    if model_key != "auto":
        model = make(model_key)
        err = _try_load(model, state, device)
        if err is None:
            return model, model_key

    model = make(guessed)
    err = _try_load(model, state, device)
    if err is None:
        return model, guessed

    if _looks_like_resnet_state(state):
        for c in ["resnet18_bn", "resnet18_gn", "resnet34_bn", "resnet34_gn"]:
            model = make(c)
            err2 = _try_load(model, state, device)
            if err2 is None:
                return model, c

    raise RuntimeError(
        f"Failed to load weights.\n"
        f"Requested={model_key}, Guessed={guessed}\n"
        f"Last error:\n{err}"
    )


# -----------------------------
# Shared: data + plotting (match occlusion style)
# -----------------------------

def clamp_adc(img: torch.Tensor, adc_lo: float, adc_hi: float) -> torch.Tensor:
    img = img.clone()
    img[img > adc_hi] = adc_hi
    img[img < adc_lo] = 0.0
    return img


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

    if info_lines:
        info_text = "\n".join(info_lines)
        fig.text(
            0.5, 0.01, info_text,
            ha="center", va="bottom",
            fontsize=18,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.savefig(out_prefix.with_name(out_prefix.name + "_map.png"), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def outputs_exist(out_prefix: Path) -> bool:
    map_png = out_prefix.with_name(out_prefix.name + "_map.png")
    meta_js = out_prefix.with_name(out_prefix.name + "_meta.txt")
    return map_png.exists() and meta_js.exists()


# def load_image_from_root(root_file: str, entry: int, plane: int, device: torch.device) -> torch.Tensor:
#     from mpid_data import mpid_data_binary
#     ds = mpid_data_binary.MPID_Dataset(root_file, "image2d_image2d_binary_tree", device.type, plane=plane)
#     x = ds[entry][0].view(1, 1, 512, 512)
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

def _rootname_simplify(root_file: str) -> str:
    root_name = Path(root_file).name
    if "dirt" in root_name:
        return "dirt"
    if "nu_overlay" in root_name:
        return "nu_overlay"
    if "offbeam" in root_name:
        return "offbeam"
    return ""


def _norm01(a: np.ndarray) -> np.ndarray:
    amin, amax = float(np.min(a)), float(np.max(a))
    if amax <= amin:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - amin) / (amax - amin)).astype(np.float32)


# -----------------------------
# Lightweight diagnostics helpers
# -----------------------------

def _active_mask_from_x(x: torch.Tensor, threshold: float = 0.0) -> np.ndarray:
    """Active pixels mask from the clamped input tensor x [1,1,H,W]."""
    img = x.detach().cpu().squeeze().numpy()
    return (img > float(threshold))


def _topk_mask(a: np.ndarray, frac: float, use_abs: bool = False) -> np.ndarray:
    """Return boolean mask of the top-|frac| pixels by value (or abs(value) if use_abs)."""
    frac = float(frac)
    if frac <= 0:
        return np.zeros_like(a, dtype=bool)
    flat = np.abs(a).ravel() if use_abs else a.ravel()
    n = flat.size
    k = int(max(1, round(frac * n)))
    k = min(k, n)
    # argpartition for speed
    idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask.reshape(a.shape)


def _overlap_topk_with_active(attr: np.ndarray, active: np.ndarray, frac: float, use_abs: bool = True) -> float:
    m = _topk_mask(attr, frac=frac, use_abs=use_abs)
    denom = float(m.sum())
    if denom == 0:
        return 0.0
    return float((m & active).sum()) / denom


def _auc_trapezoid(y, x):
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.size < 2:
        return float('nan')
    return float(np.trapezoid(y, x))


def _deletion_insertion_curves(
    model: nn.Module,
    x: torch.Tensor,
    attr: np.ndarray,
    signal_idx: int = 0,
    background_idx: int = 1,
    n_steps: int = 11,
    seed: int = 123,
):
    """Deletion/insertion curves using per-pixel ranking by |attr|.

    Deletion: progressively set top attributed pixels to baseline (zeros).
    Insertion: progressively reveal top attributed pixels starting from baseline.

    Returns a dict with curves and AUCs for signal/background, plus a random baseline.
    """
    rng = np.random.default_rng(int(seed))
    H, W = attr.shape
    n = H * W
    flat = np.abs(attr).reshape(-1)
    order = np.argsort(flat)[::-1]
    rand_order = rng.permutation(n)

    fracs = np.linspace(0.0, 1.0, int(n_steps))

    baseline = torch.zeros_like(x)
    out = {
        "fractions": fracs.tolist(),
        "signal": {"deletion": [], "insertion": [], "random_deletion": [], "random_insertion": []},
        "background": {"deletion": [], "insertion": [], "random_deletion": [], "random_insertion": []},
    }

    x0 = x

    with torch.no_grad():
        for frac in fracs:
            k = int(round(frac * n))
            k = min(max(k, 0), n)

            m = np.zeros(n, dtype=bool)
            if k > 0:
                m[order[:k]] = True
            m = m.reshape(H, W)
            m_t = torch.from_numpy(m).to(x0.device)
            m_t = m_t.view(1, 1, H, W)

            mr = np.zeros(n, dtype=bool)
            if k > 0:
                mr[rand_order[:k]] = True
            mr = mr.reshape(H, W)
            mr_t = torch.from_numpy(mr).to(x0.device).view(1, 1, H, W)

            # deletion
            x_del = x0.clone()
            x_del[m_t] = baseline[m_t]
            logits = model(x_del)
            probs = torch.sigmoid(logits)[0]
            out["signal"]["deletion"].append(float(probs[signal_idx].item()))
            out["background"]["deletion"].append(float(probs[background_idx].item()))

            x_del_r = x0.clone()
            x_del_r[mr_t] = baseline[mr_t]
            logits_r = model(x_del_r)
            probs_r = torch.sigmoid(logits_r)[0]
            out["signal"]["random_deletion"].append(float(probs_r[signal_idx].item()))
            out["background"]["random_deletion"].append(float(probs_r[background_idx].item()))

            # insertion
            x_ins = baseline + (x0 - baseline) * m_t.float()
            logits = model(x_ins)
            probs = torch.sigmoid(logits)[0]
            out["signal"]["insertion"].append(float(probs[signal_idx].item()))
            out["background"]["insertion"].append(float(probs[background_idx].item()))

            x_ins_r = baseline + (x0 - baseline) * mr_t.float()
            logits_r = model(x_ins_r)
            probs_r = torch.sigmoid(logits_r)[0]
            out["signal"]["random_insertion"].append(float(probs_r[signal_idx].item()))
            out["background"]["random_insertion"].append(float(probs_r[background_idx].item()))

    # AUCs (useful scalars)
    out["signal"]["auc_deletion"] = _auc_trapezoid(out["signal"]["deletion"], fracs)
    out["signal"]["auc_insertion"] = _auc_trapezoid(out["signal"]["insertion"], fracs)
    out["signal"]["auc_random_deletion"] = _auc_trapezoid(out["signal"]["random_deletion"], fracs)
    out["signal"]["auc_random_insertion"] = _auc_trapezoid(out["signal"]["random_insertion"], fracs)

    out["background"]["auc_deletion"] = _auc_trapezoid(out["background"]["deletion"], fracs)
    out["background"]["auc_insertion"] = _auc_trapezoid(out["background"]["insertion"], fracs)
    out["background"]["auc_random_deletion"] = _auc_trapezoid(out["background"]["random_deletion"], fracs)
    out["background"]["auc_random_insertion"] = _auc_trapezoid(out["background"]["random_insertion"], fracs)

    return out


# -----------------------------
# Integrated Gradients (two-class maps)
# -----------------------------

def integrated_gradients(model: nn.Module, x: torch.Tensor, target_idx: int, steps: int = 64) -> torch.Tensor:
    baseline = torch.zeros_like(x)

    grads = []
    for i in range(1, int(steps) + 1):
        alpha = float(i) / float(steps)
        xi = baseline + alpha * (x - baseline)
        xi = xi.clone().detach().requires_grad_(True)

        logits = model(xi)
        score = logits[:, target_idx].sum()

        model.zero_grad(set_to_none=True)
        if xi.grad is not None:
            xi.grad.zero_()
        score.backward()
        grads.append(xi.grad.detach())

    avg_grad = torch.mean(torch.stack(grads, dim=0), dim=0)
    ig = (x - baseline) * avg_grad
    return ig


def ig_two_class_maps(
    model: nn.Module,
    x: torch.Tensor,
    steps: int,
    positive_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    ig_sig = integrated_gradients(model, x, target_idx=0, steps=steps)[0, 0].detach().cpu().numpy()
    ig_bkg = integrated_gradients(model, x, target_idx=1, steps=steps)[0, 0].detach().cpu().numpy()

    if positive_only:
        ig_sig = np.maximum(ig_sig, 0.0)
        ig_bkg = np.maximum(ig_bkg, 0.0)

    return ig_sig.astype(np.float32), ig_bkg.astype(np.float32)


def run_one(
    model: nn.Module,
    model_key: str,
    weight_file: str,
    root_file: str,
    entry: int,
    n_pixels: Optional[int],
    out_dir: str,
    tag: str,
    steps: int,
    signed: bool,
    active_threshold: float,
    topk_frac: float,
    diag_curves: bool,
    curve_steps: int,
    curve_seed: int,
    adc_lo: float,
    adc_hi: float,
    normalize: bool,
    plane: int,
    save_npy: bool,
    device: torch.device,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root_name = _rootname_simplify(root_file)
    out_prefix = out_dir / (f"integrad__ENTRY_{entry}__{root_name}" if root_name else f"integrad__ENTRY_{entry}")

    if outputs_exist(out_prefix):
        print(f"[skip] exists: {out_prefix}")
        return

    x = load_image_from_root(root_file, entry, plane, device)
    x = x.to(device)
    x = clamp_adc(x, adc_lo, adc_hi)
    print("x_clamped fingerprint:", tensor_fingerprint(x))

    with torch.no_grad():
        logits0 = model(x)
        probs0 = torch.sigmoid(logits0).detach().cpu().numpy()[0]
        base_sig = float(probs0[0])
        base_bkg = float(probs0[1])

    print("\n--- DIAG forward ---")
    print(score_summary_from_logits(logits0))
    print("--------------------\n")
    
    # --- Integrated Gradients ---
    # Compute raw (signed) IG for diagnostics (completeness / curves).
    ig_sig_t = integrated_gradients(model, x, target_idx=0, steps=steps)
    ig_bkg_t = integrated_gradients(model, x, target_idx=1, steps=steps)
    ig_sig_raw = ig_sig_t[0, 0].detach().cpu().numpy().astype(np.float32)
    ig_bkg_raw = ig_bkg_t[0, 0].detach().cpu().numpy().astype(np.float32)
    # For plotting, keep previous default behaviour (positive-only) unless --signed is set.
    positive_only = (not bool(signed))
    sig_map = ig_sig_raw.copy()
    bkg_map = ig_bkg_raw.copy()
    if positive_only:
        sig_map = np.maximum(sig_map, 0.0)
        bkg_map = np.maximum(bkg_map, 0.0)

    if normalize:
        sig_map = _norm01(sig_map)
        bkg_map = _norm01(bkg_map)

    original_img = x.detach().cpu().squeeze().numpy().astype(np.float32)
    active = _active_mask_from_x(x, threshold=active_threshold)

    # --- Diagnostics ---
    # Completeness: sum(IG) ~= logit(x) - logit(baseline)
    with torch.no_grad():
        baseline = torch.zeros_like(x)
        logits_x = model(x)[0]
        logits_b = model(baseline)[0]
    diag = {
        "active_frac": float(active.mean()),
        "ig_completeness": {},
        "overlap": {},
    }
    for idx, name, ig_t, ig_raw in [
        (0, "signal", ig_sig_t, ig_sig_raw),
        (1, "background", ig_bkg_t, ig_bkg_raw),
    ]:
        delta = float((logits_x[idx] - logits_b[idx]).item())
        ig_sum = float(ig_t.sum().item())
        rel_err = abs(ig_sum - delta) / (abs(delta) + 1e-8)
        diag["ig_completeness"][name] = {
            "delta_logit": delta,
            "ig_sum": ig_sum,
            "rel_error": float(rel_err),
        }

    diag["overlap"]["signal_topk_on_active"] = _overlap_topk_with_active(ig_sig_raw, active, frac=topk_frac, use_abs=True)
    diag["overlap"]["background_topk_on_active"] = _overlap_topk_with_active(ig_bkg_raw, active, frac=topk_frac, use_abs=True)

    if diag_curves:
        diag["curves"] = {
            "signal": _deletion_insertion_curves(model, x, ig_sig_raw, signal_idx=0, background_idx=1, n_steps=curve_steps, seed=curve_seed),
            "background": _deletion_insertion_curves(model, x, ig_bkg_raw, signal_idx=0, background_idx=1, n_steps=curve_steps, seed=curve_seed),
        }

    save_combined_map_png(
        signal_arr=sig_map,
        background_arr=bkg_map,
        original_img=original_img,
        out_prefix=out_prefix,
        entry_number=entry,
        n_pixels=n_pixels,
        base_signal_score=base_sig,
        base_background_score=base_bkg,
        tag=tag,
        model_key=model_key,
        method="integrated_gradients",
    )

    meta = {
        "model": model_key,
        "weight_file": str(weight_file),
        "input_file": str(root_file),
        "entry_number": int(entry),
        "n_pixels": None if n_pixels is None else int(n_pixels),
        "base_signal_score": float(base_sig),
        "base_background_score": float(base_bkg),
        "method": "integrated_gradients",
        "steps": int(steps),
        "adc_lo": float(adc_lo),
        "adc_hi": float(adc_hi),
        "normalize": bool(normalize),
        "plane": int(plane),
        "reason": tag,
        "diagnostics": diag,
    }

    meta_path = out_prefix.with_name(out_prefix.name + "_meta.txt")
    meta_path.write_text(json.dumps(meta, indent=2))
    print("[ok]", meta_path)

    if save_npy:
        np.save(out_prefix.with_name(out_prefix.name + "_Signal_map.npy"), sig_map)
        np.save(out_prefix.with_name(out_prefix.name + "_Background_map.npy"), bkg_map)


# -----------------------------
# CLI (match occlusion style)
# -----------------------------

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="auto", help="auto | mpid | resnet18_bn | resnet18_gn | resnet34_bn | resnet34_gn")
    p.add_argument("--weight-file", required=True)

    p.add_argument("--input-file", default=None, help="Path to ROOT file (inside container: /data/...)")
    p.add_argument("--entry", type=int, default=None, help="Single entry index in ROOT tree")
    p.add_argument("--entries", type=int, default=1, help="How many entries to run starting at --entry")

    p.add_argument("--from-csv", default=None, help="CSV with columns: root_file, entry_number, n_pixels, out_dir, tag, weight_file(optional), model(optional)")
    p.add_argument("--larcv-base", default="/data", help="Base bind inside container for ROOTs (default /data)")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--tag", default="")

    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--adc-lo", type=float, default=10.0)
    p.add_argument("--adc-hi", type=float, default=500.0)

    # In occlusion, --normalize means normalize the produced maps (min-max).
    p.add_argument("--normalize", action="store_true")

    p.add_argument("--signed", action="store_true", help="If set, do NOT clamp IG to positive only (keeps signed attributions).")
    p.add_argument("--active-threshold", type=float, default=0.0, help="Threshold on clamped ADC to define active pixels for overlap diagnostics.")
    p.add_argument("--topk-frac", type=float, default=0.01, help="Top fraction of pixels for overlap / curves (default 1%).")
    p.add_argument("--diag-curves", action="store_true", help="Run deletion/insertion curves (extra forward passes).")
    p.add_argument("--curve-steps", type=int, default=11, help="Number of points in deletion/insertion curves.")
    p.add_argument("--curve-seed", type=int, default=123, help="Random seed for random baseline curves.")

    p.add_argument("--plane", type=int, default=0)
    p.add_argument("--save-npy", action="store_true")
    p.add_argument("--gpuid", default="0")
    p.add_argument("--n-pixels", type=int, default=None)
    return p



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
            if not os.path.isabs(root_file) and args.larcv_base:
                root_file = str(Path(args.larcv_base) / root_file)

            entry = int(row["entry_number"])
            n_pixels = int(row["n_pixels"])
            out_dir = str(row["out_dir"])
            tag = str(row["tag"])

            wfile = str(row["weight_file"]) if ("weight_file" in df.columns and isinstance(row["weight_file"], str)) else args.weight_file
            mkey = str(row["model"]) if ("model" in df.columns and isinstance(row["model"], str)) else args.model

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
                steps=args.steps,
                signed=args.signed,
                active_threshold=args.active_threshold,
                topk_frac=args.topk_frac,
                diag_curves=args.diag_curves,
                curve_steps=args.curve_steps,
                curve_seed=args.curve_seed,
                adc_lo=args.adc_lo,
                adc_hi=args.adc_hi,
                normalize=args.normalize,
                plane=args.plane,
                save_npy=args.save_npy,
                device=device,
            )
        return

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
            steps=args.steps,
            signed=args.signed,
            active_threshold=args.active_threshold,
            topk_frac=args.topk_frac,
            diag_curves=args.diag_curves,
            curve_steps=args.curve_steps,
            curve_seed=args.curve_seed,
            adc_lo=args.adc_lo,
            adc_hi=args.adc_hi,
            normalize=args.normalize,
            plane=args.plane,
            save_npy=args.save_npy,
            device=device,
        )


if __name__ == "__main__":
    main()
