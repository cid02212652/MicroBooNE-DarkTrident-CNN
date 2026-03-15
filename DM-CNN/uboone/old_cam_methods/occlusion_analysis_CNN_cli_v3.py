#!/usr/bin/env python3
"""Occlusion analysis integrated into the shared xai_common framework.

Key differences vs original occlusion script:
- Uses xai_common.build_model + shared IO/plotting/diagnostics for consistency.
- Produces an *attribution* map defined as:  attr(x,y) = base_prob - prob_with_patch_occluded
  (so larger values mean the occluded patch was more important for that class).
- Uses the same overlap + deletion/insertion diagnostics schema as the other v3 scripts.

Original script reference: occlusion_analysis_CNN_cli.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from xai_common import (
    active_mask_from_x,
    build_model,
    clamp_adc,
    deletion_insertion_curves,
    gn_summary,
    load_image_from_root,
    norm01,
    overlap_topk_on_active,
    outputs_exist,
    rootname_simplify,
    save_combined_map_png,
    score_summary_from_logits,
    sha256_file,
    tensor_fingerprint,
)


# -----------------------------
# Occlusion logic
# -----------------------------

@torch.no_grad()
def occlusion_attribution_maps(
    model: nn.Module,
    x: torch.Tensor,  # [1,1,H,W] on device
    occlusion_size: int = 4,
    stride: int = 1,
    adc_lo: float = 10.0,
    adc_hi: float = 500.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, object]:
    """Return occlusion-based attribution maps for both classes.

    Attribution definition (per pixel center):
      attr = base_prob - prob( x with (2*occlusion_size+1)^2 patch zeroed )

    Unscanned pixels (inactive centers or borders) remain 0 attribution.
    """

    model.eval()

    # Clamp like other scripts
    x0 = clamp_adc(x, adc_lo, adc_hi)

    H, W = int(x0.shape[-2]), int(x0.shape[-1])
    pad = int(occlusion_size)

    logits0 = model(x0)
    probs0 = torch.sigmoid(logits0)[0].detach().cpu().numpy()
    base_sig, base_bkg = float(probs0[0]), float(probs0[1])

    # We store *occluded* probabilities at each scanned center, defaulting to base
    occ_sig = np.full((H, W), base_sig, dtype=np.float32)
    occ_bkg = np.full((H, W), base_bkg, dtype=np.float32)

    # Avoid per-iteration .item() device sync by precomputing a CPU active mask.
    # (This matches the original behavior of skipping centers with 0 ADC after clamping.)
    active_centers = (x0[0, 0].detach().cpu().numpy() != 0.0)

    # Scan over valid centers (avoid borders where the patch would spill out)
    for i in range(pad, H - pad, int(stride)):
        for j in range(pad, W - pad, int(stride)):
            if not active_centers[i, j]:
                continue

            x_occ = x0.clone()
            x_occ[0, 0, i - pad : i + pad + 1, j - pad : j + pad + 1] = 0.0

            logits = model(x_occ)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

            occ_sig[i, j] = float(probs[0])
            occ_bkg[i, j] = float(probs[1])

    # Convert to attribution (drop in prob when occluded)
    attr_sig = (base_sig - occ_sig).astype(np.float32)
    attr_bkg = (base_bkg - occ_bkg).astype(np.float32)

    return {
        "attr_signal": attr_sig,
        "attr_background": attr_bkg,
        "occ_signal": occ_sig,
        "occ_background": occ_bkg,
        "base_signal_score": base_sig,
        "base_background_score": base_bkg,
        "logits0": logits0,
    }


# -----------------------------
# Run one
# -----------------------------

def run_one(
    model: nn.Module,
    model_key: str,
    weight_file: str,
    root_file: str,
    entry: int,
    n_pixels: Optional[int],
    out_dir: str,
    tag: str,
    occlusion_size: int,
    stride: int,
    adc_lo: float,
    adc_hi: float,
    normalize: bool,
    plane: int,
    save_npy: bool,
    active_threshold: float,
    topk_frac: float,
    diag_curves: bool,
    curve_steps: int,
    curve_seed: int,
    device: torch.device,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root_tag = rootname_simplify(root_file)
    out_prefix = out_dir / (f"occlusion__ENTRY_{entry}__{root_tag}" if root_tag else f"occlusion__ENTRY_{entry}")

    if outputs_exist(out_prefix, layer_name=None):
        print(f"[skip] exists: {out_prefix}")
        return

    x = load_image_from_root(root_file, entry, plane, device, diag_print=True).to(device)
    x = clamp_adc(x, adc_lo, adc_hi)
    print("x_clamped fingerprint:", tensor_fingerprint(x))

    res = occlusion_attribution_maps(
        model=model,
        x=x,
        occlusion_size=int(occlusion_size),
        stride=int(stride),
        adc_lo=float(adc_lo),
        adc_hi=float(adc_hi),
        device=device,
    )

    logits0 = res["logits0"]
    print("\n--- DIAG forward ---")
    print(score_summary_from_logits(logits0))
    print("--------------------\n")

    sig_map = res["attr_signal"]
    bkg_map = res["attr_background"]

    # Optional map normalization for visualization
    if normalize:
        sig_map = norm01(sig_map)
        bkg_map = norm01(bkg_map)

    original_img = x.detach().cpu().squeeze().numpy().astype(np.float32)

    # --- Diagnostics: consistent schema used by all scripts ---
    diagnostics = {
        "active_threshold": float(active_threshold),
        "topk_frac": float(topk_frac),
        "occlusion_attribution": "delta_prob",  # base_prob - occluded_prob
        "occlusion_patch_halfwidth": int(occlusion_size),
        "stride": int(stride),
    }

    active = active_mask_from_x(x, threshold=active_threshold)
    diagnostics["active_frac"] = float(active.mean())

    diagnostics["overlap_topk_signal"] = overlap_topk_on_active(sig_map, active, topk_frac, use_abs=True)
    diagnostics["overlap_topk_background"] = overlap_topk_on_active(bkg_map, active, topk_frac, use_abs=True)

    if diag_curves:
        diagnostics["curves"] = deletion_insertion_curves(
            model=model,
            x=x,
            attr_rank=sig_map,          # consistent choice: order by signal attribution map
            signal_idx=0,
            background_idx=1,
            n_steps=int(curve_steps),
            seed=int(curve_seed),
            baseline=torch.zeros_like(x),
            use_abs_rank=True,
        )
        diagnostics["curves_attr_used"] = "occlusion_attr_signal"

    # Note: plot labels in save_combined_map_png are generic; these are attribution (delta) maps.
    extra = f"attr=base_prob-occluded_prob; patch={(2*int(occlusion_size)+1)}x{(2*int(occlusion_size)+1)}; stride={int(stride)}"
    save_combined_map_png(
        signal_arr=sig_map,
        background_arr=bkg_map,
        original_img=original_img,
        out_prefix=out_prefix,
        entry_number=int(entry),
        n_pixels=(None if n_pixels is None else int(n_pixels)),
        base_signal_score=float(res["base_signal_score"]),
        base_background_score=float(res["base_background_score"]),
        tag=tag,
        model_key=model_key,
        method="occlusion",
        layer_name=None,
        extra_info=extra,
    )

    meta = {
        "model": model_key,
        "weight_file": str(weight_file),
        "input_file": str(root_file),
        "entry_number": int(entry),
        "n_pixels": None if n_pixels is None else int(n_pixels),
        "base_signal_score": float(res["base_signal_score"]),
        "base_background_score": float(res["base_background_score"]),
        "method": "occlusion",
        "occlusion_size": int(occlusion_size),
        "stride": int(stride),
        "adc_lo": float(adc_lo),
        "adc_hi": float(adc_hi),
        "normalize": bool(normalize),
        "plane": int(plane),
        "reason": str(tag),
        "diagnostics": diagnostics,
    }

    meta_path = out_prefix.with_name(out_prefix.name + "_meta.txt")
    meta_path.write_text(json.dumps(meta, indent=2))
    print("[ok]", meta_path)

    if save_npy:
        np.save(out_prefix.with_name(out_prefix.name + "_Signal_map.npy"), sig_map)
        np.save(out_prefix.with_name(out_prefix.name + "_Background_map.npy"), bkg_map)


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

    p.add_argument(
        "--from-csv",
        default=None,
        help="CSV with columns: root_file, entry_number, n_pixels, out_dir, tag (plus optional weight_file, model)",
    )
    p.add_argument("--larcv-base", default="/data", help="Base bind inside container for ROOTs (default /data)")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--tag", default="")

    p.add_argument("--occlusion-size", type=int, default=4, help="Half-width of square patch; patch is (2*size+1)^2")
    p.add_argument("--stride", type=int, default=1)

    # Diagnostics
    p.add_argument("--active-threshold", type=float, default=0.0)
    p.add_argument("--topk-frac", type=float, default=0.01)
    p.add_argument("--diag-curves", action="store_true")
    p.add_argument("--curve-steps", type=int, default=11)
    p.add_argument("--curve-seed", type=int, default=123)

    p.add_argument("--adc-lo", type=float, default=10.0)
    p.add_argument("--adc-hi", type=float, default=500.0)
    p.add_argument("--normalize", action="store_true", help="Normalize output attribution maps to [0,1] for plotting")

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
            n_pixels = None
            try:
                n_pixels = int(row["n_pixels"]) if not (row["n_pixels"] is None or (isinstance(row["n_pixels"], float) and np.isnan(row["n_pixels"]))) else None
            except Exception:
                n_pixels = None

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
                occlusion_size=args.occlusion_size,
                stride=args.stride,
                adc_lo=args.adc_lo,
                adc_hi=args.adc_hi,
                normalize=args.normalize,
                plane=args.plane,
                save_npy=args.save_npy,
                active_threshold=args.active_threshold,
                topk_frac=args.topk_frac,
                diag_curves=args.diag_curves,
                curve_steps=args.curve_steps,
                curve_seed=args.curve_seed,
                device=device,
            )
        return

    if args.input_file is None or args.entry is None:
        raise SystemExit("Need --input-file and --entry unless using --from-csv")

    for e in range(int(args.entry), int(args.entry) + int(args.entries)):
        run_one(
            model=model,
            model_key=resolved_key,
            weight_file=args.weight_file,
            root_file=args.input_file,
            entry=int(e),
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
            active_threshold=args.active_threshold,
            topk_frac=args.topk_frac,
            diag_curves=args.diag_curves,
            curve_steps=args.curve_steps,
            curve_seed=args.curve_seed,
            device=device,
        )


if __name__ == "__main__":
    main()
