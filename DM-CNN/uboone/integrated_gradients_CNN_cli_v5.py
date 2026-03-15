#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from xai_common import (
    build_model,
    clamp_adc,
    gn_summary,
    load_image_from_root,
    overlap_topk_on_active,
    outputs_exist,
    rootname_simplify,
    save_combined_map_png,
    score_summary_from_logits,
    sha256_file,
    tensor_fingerprint,
    active_mask_from_x,
    deletion_insertion_curves,
    norm01,
    mask_diagnostics,
    curves_pixel_need_summary,
    infer_mass_from_filename,
    attr_adc_correlation,
)


# -----------------------------
# Integrated Gradients
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
    return (x - baseline) * avg_grad


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

    root_tag = rootname_simplify(root_file)
    out_prefix = out_dir / (f"integrad__ENTRY_{entry}__{root_tag}" if root_tag else f"integrad__ENTRY_{entry}")

    if outputs_exist(out_prefix, layer_name=None):
        print(f"[skip] exists: {out_prefix}")
        return

    x = load_image_from_root(root_file, entry, plane, device, diag_print=True).to(device)
    x = clamp_adc(x, adc_lo, adc_hi)
    print("x_clamped fingerprint:", tensor_fingerprint(x))

    with torch.no_grad():
        logits0 = model(x)
        probs0 = torch.sigmoid(logits0)[0].detach().cpu().numpy()
        base_sig, base_bkg = float(probs0[0]), float(probs0[1])

    print("\n--- DIAG forward ---")
    print(score_summary_from_logits(logits0))
    print("--------------------\n")

    # Raw (signed) IG for diagnostics
    ig_sig_t = integrated_gradients(model, x, target_idx=0, steps=steps)
    ig_bkg_t = integrated_gradients(model, x, target_idx=1, steps=steps)
    ig_sig_raw = ig_sig_t[0, 0].detach().cpu().numpy().astype(np.float32)
    ig_bkg_raw = ig_bkg_t[0, 0].detach().cpu().numpy().astype(np.float32)

    # Keep raw maps for diagnostics and .npy; apply optional norm01 only for PNG display
    sig_map_raw = ig_sig_raw.copy()
    bkg_map_raw = ig_bkg_raw.copy()
    if not signed:
        sig_map_raw = np.maximum(sig_map_raw, 0.0)
        bkg_map_raw = np.maximum(bkg_map_raw, 0.0)

    sig_map_plot = norm01(sig_map_raw) if normalize else sig_map_raw
    bkg_map_plot = norm01(bkg_map_raw) if normalize else bkg_map_raw

    original_img = x.detach().cpu().squeeze().numpy().astype(np.float32)
    mass_token = infer_mass_from_filename(root_file)

    # --- Diagnostics: consistent keys across scripts ---
    diagnostics = {
        "active_threshold": float(active_threshold),
        "topk_frac": float(topk_frac),
        "signed_ig_used_for_diagnostics": True,
        "ig_completeness": {},
    }

    active = active_mask_from_x(x, threshold=active_threshold)
    diagnostics["active_frac"] = float(active.mean())

    # Completeness: sum(IG) ~= logit(x) - logit(baseline)
    with torch.no_grad():
        baseline = torch.zeros_like(x)
        logits_x = model(x)[0]
        logits_b = model(baseline)[0]

    for idx, name, ig_t in [
        (0, "signal", ig_sig_t),
        (1, "background", ig_bkg_t),
    ]:
        delta = float((logits_x[idx] - logits_b[idx]).item())
        ig_sum = float(ig_t.sum().item())
        rel_err = abs(ig_sum - delta) / (abs(delta) + 1e-8)
        diagnostics["ig_completeness"][name] = {
            "delta_logit": delta,
            "ig_sum": ig_sum,
            "rel_error": float(rel_err),
        }

    diagnostics["overlap_topk_signal"] = overlap_topk_on_active(ig_sig_raw, active, topk_frac, use_abs=True)
    diagnostics["overlap_topk_background"] = overlap_topk_on_active(ig_bkg_raw, active, topk_frac, use_abs=True)

    # Richer, less-arbitrary mask diagnostics at multiple fractions
    diagnostics["mask_diagnostics"] = {
        "signal": mask_diagnostics(sig_map_raw, active, use_abs=True),
        "background": mask_diagnostics(bkg_map_raw, active, use_abs=True),
    }

    diagnostics["attr_adc_correlation"] = {
        "signal": attr_adc_correlation(sig_map_raw, original_img, active, use_abs=True),
        "background": attr_adc_correlation(bkg_map_raw, original_img, active, use_abs=True),
    }


    if diag_curves:
        diagnostics["curves"] = deletion_insertion_curves(
            model=model,
            x=x,
            attr_rank=ig_sig_raw,       # consistent choice: order by signal attribution map
            signal_idx=0,
            background_idx=1,
            n_steps=int(curve_steps),
            seed=int(curve_seed),
            baseline=torch.zeros_like(x),
            use_abs_rank=True,
        )
        diagnostics["curves_attr_used"] = "ig_signal_raw"
        diagnostics["curves_pixel_need"] = curves_pixel_need_summary(diagnostics["curves"])

    save_combined_map_png(
        signal_arr=sig_map_plot,
        background_arr=bkg_map_plot,
        original_img=original_img,
        out_prefix=out_prefix,
        entry_number=entry,
        n_pixels=n_pixels,
        base_signal_score=base_sig,
        base_background_score=base_bkg,
        tag=tag,
        model_key=model_key,
        method="integrated_gradients",
        layer_name=None,
    )

    meta = {
        "model": model_key,
        "weight_file": str(weight_file),
        "input_file": str(root_file),
        "mass_token": (None if mass_token is None else str(mass_token)),
        "entry_number": int(entry),
        "n_pixels": None if n_pixels is None else int(n_pixels),
        "base_signal_score": float(base_sig),
        "base_background_score": float(base_bkg),
        "method": "integrated_gradients",
        "steps": int(steps),
        "signed": bool(signed),
        "adc_lo": float(adc_lo),
        "adc_hi": float(adc_hi),
        "normalize_for_png": bool(normalize),
        "saved_npy_is_raw": True,
        "plane": int(plane),
        "reason": tag,
        "diagnostics": diagnostics,
    }

    meta_path = out_prefix.with_name(out_prefix.name + "_meta.txt")
    meta_path.write_text(json.dumps(meta, indent=2))
    print("[ok]", meta_path)

    if save_npy:
        np.save(out_prefix.with_name(out_prefix.name + "_input.npy"), original_img)
        np.save(out_prefix.with_name(out_prefix.name + "_signal_map.npy"), sig_map_raw)
        np.save(out_prefix.with_name(out_prefix.name + "_background_map.npy"), bkg_map_raw)


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

    p.add_argument("--from-csv", default=None, help="CSV with columns: root_file, entry_number, n_pixels, out_dir, tag, weight_file(optional), model(optional)")
    p.add_argument("--larcv-base", default="/data", help="Base bind inside container for ROOTs (default /data)")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--tag", default="")

    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--signed", action="store_true", help="If set, do NOT clamp IG to positive only for plotting.")

    # Diagnostics
    p.add_argument("--active-threshold", type=float, default=0.0)
    p.add_argument("--topk-frac", type=float, default=0.01)
    p.add_argument("--diag-curves", action="store_true")
    p.add_argument("--curve-steps", type=int, default=11)
    p.add_argument("--curve-seed", type=int, default=123)

    p.add_argument("--adc-lo", type=float, default=10.0)
    p.add_argument("--adc-hi", type=float, default=500.0)
    p.add_argument("--normalize", action="store_true")
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
            entry=int(e),
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
