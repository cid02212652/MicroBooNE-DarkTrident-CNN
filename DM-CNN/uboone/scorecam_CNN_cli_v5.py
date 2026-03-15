#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xai_common import (
    active_mask_from_x,
    build_model,
    clamp_adc,
    deletion_insertion_curves,
    get_module_by_name,
    gn_summary,
    load_image_from_root,
    norm01,
    overlap_topk_on_active,
    outputs_exist,
    preset_layer_name,
    rootname_simplify,
    save_combined_map_png,
    sha256_file,
    mask_diagnostics,
    curves_pixel_need_summary,
    infer_mass_from_filename,
    attr_adc_correlation,
)


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


@torch.no_grad()
def scorecam_maps(
    model: nn.Module,
    x: torch.Tensor,  # [1,1,H,W]
    layer: nn.Module,
    max_channels: int = 64,
    batch_size: int = 16,
    channel_mode: str = "topk",
    use_relu_acts: bool = True,
    upsample_mode: str = "nearest",
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (signal_map, background_map) at input resolution as float32 numpy arrays."""
    model.eval()
    H, W = x.shape[-2], x.shape[-1]

    # capture activations
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

    # choose channels
    energy = torch.mean(torch.abs(acts), dim=(1, 2))  # [C]
    k = min(int(max_channels), int(C))
    if channel_mode == "topk":
        idx = torch.topk(energy, k=k, largest=True).indices
    elif channel_mode == "first":
        idx = torch.arange(k, device=acts.device)
    else:
        raise ValueError("--scorecam-channel-mode must be 'topk' or 'first'.")

    cam_sig = torch.zeros((h, w), device=acts.device, dtype=acts.dtype)
    cam_bkg = torch.zeros((h, w), device=acts.device, dtype=acts.dtype)

    for s in range(0, idx.numel(), int(batch_size)):
        ii = idx[s : s + int(batch_size)]
        a = acts[ii]  # [B,h,w]
        B = a.shape[0]

        if str(upsample_mode).lower() == "nearest":
            masks = F.interpolate(a.unsqueeze(1), size=(H, W), mode="nearest")  # [B,1,H,W]
        else:
            masks = F.interpolate(a.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)  # [B,1,H,W]  # [B,1,H,W]
        m_min = masks.amin(dim=(2, 3), keepdim=True)
        masks = masks - m_min
        m_max = masks.amax(dim=(2, 3), keepdim=True)
        masks = masks / (m_max + eps)

        xb = x.repeat(B, 1, 1, 1) * masks  # [B,1,H,W]
        probs = torch.sigmoid(model(xb))    # [B,2]

        w_sig = probs[:, 0].view(B, 1, 1)
        w_bkg = probs[:, 1].view(B, 1, 1)

        cam_sig += torch.sum(w_sig * a, dim=0)
        cam_bkg += torch.sum(w_bkg * a, dim=0)

    cam_sig = torch.relu(cam_sig)
    cam_bkg = torch.relu(cam_bkg)

    if str(upsample_mode).lower() == "nearest":
        cam_sig_up = F.interpolate(cam_sig.view(1, 1, h, w), size=(H, W), mode="nearest")[0, 0]
        cam_bkg_up = F.interpolate(cam_bkg.view(1, 1, h, w), size=(H, W), mode="nearest")[0, 0]
    else:
        cam_sig_up = F.interpolate(cam_sig.view(1, 1, h, w), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        cam_bkg_up = F.interpolate(cam_bkg.view(1, 1, h, w), size=(H, W), mode="bilinear", align_corners=False)[0, 0]

    return cam_sig_up.detach().cpu().numpy().astype(np.float32), cam_bkg_up.detach().cpu().numpy().astype(np.float32)


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
    adc_lo: float,
    adc_hi: float,
    normalize: bool,
    plane: int,
    save_npy: bool,
    layer_name: Optional[str],
    layer_preset: str,
    active_threshold: float,
    topk_frac: float,
    diag_curves: bool,
    curve_steps: int,
    curve_seed: int,
    scorecam_max_channels: int,
    scorecam_batch: int,
    scorecam_channel_mode: str,
    scorecam_use_relu_acts: bool,
    cam_upsample: str,
    device: torch.device,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if layer_name is None:
        layer_name = preset_layer_name(model, preset=layer_preset)

    root_tag = rootname_simplify(root_file)
    out_prefix = out_dir / (f"scorecam__ENTRY_{entry}__{root_tag}" if root_tag else f"scorecam__ENTRY_{entry}")

    if outputs_exist(out_prefix, layer_name):
        print(f"[skip] exists: {out_prefix} layer={layer_name}")
        return

    x = load_image_from_root(root_file, entry, plane, device, diag_print=False).to(device)
    x = clamp_adc(x, adc_lo, adc_hi)

    with torch.no_grad():
        logits0 = model(x)
        probs0 = torch.sigmoid(logits0)[0].detach().cpu().numpy()
    base_sig, base_bkg = float(probs0[0]), float(probs0[1])

    layer = get_module_by_name(model, layer_name)

    sig_map, bkg_map = scorecam_maps(
        model=model,
        x=x,
        layer=layer,
        max_channels=int(scorecam_max_channels),
        batch_size=int(scorecam_batch),
        channel_mode=str(scorecam_channel_mode),
        use_relu_acts=bool(scorecam_use_relu_acts),
        upsample_mode=str(cam_upsample),
    )

    # Keep raw maps for diagnostics and .npy; apply optional norm01 only for PNG display
    sig_map_raw = sig_map
    bkg_map_raw = bkg_map
    sig_map_plot = norm01(sig_map_raw) if normalize else sig_map_raw
    bkg_map_plot = norm01(bkg_map_raw) if normalize else bkg_map_raw

    original_img = x.detach().cpu().squeeze().numpy().astype(np.float32)
    mass_token = infer_mass_from_filename(root_file)

    diagnostics = {
        "active_threshold": float(active_threshold),
        "topk_frac": float(topk_frac),
    }
    active = active_mask_from_x(x, threshold=active_threshold)
    diagnostics["active_frac"] = float(active.mean())
    diagnostics["overlap_topk_signal"] = overlap_topk_on_active(sig_map_raw, active, topk_frac, use_abs=True)
    diagnostics["overlap_topk_background"] = overlap_topk_on_active(bkg_map_raw, active, topk_frac, use_abs=True)

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
            attr_rank=sig_map_raw,      # consistent choice: order by signal attribution map
            signal_idx=0,
            background_idx=1,
            n_steps=int(curve_steps),
            seed=int(curve_seed),
            baseline=torch.zeros_like(x),
            use_abs_rank=True,
        )
        diagnostics["curves_attr_used"] = "signal_map"
        diagnostics["curves_pixel_need"] = curves_pixel_need_summary(diagnostics["curves"])

    extra = f"Channels <= {int(scorecam_max_channels)}; batch={int(scorecam_batch)}; mode={scorecam_channel_mode}; upsample={cam_upsample}"
    save_combined_map_png(
        signal_arr=sig_map_plot,
        background_arr=bkg_map_plot,
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
        "mass_token": (None if mass_token is None else str(mass_token)),
        "entry_number": int(entry),
        "n_pixels": (None if n_pixels is None else int(n_pixels)),
        "base_signal_score": float(base_sig),
        "base_background_score": float(base_bkg),
        "layer_name": str(layer_name),
        "scorecam_max_channels": int(scorecam_max_channels),
        "scorecam_batch": int(scorecam_batch),
        "scorecam_channel_mode": str(scorecam_channel_mode),
        "scorecam_use_relu_acts": bool(scorecam_use_relu_acts),
        "cam_upsample": str(cam_upsample),
        "adc_lo": float(adc_lo),
        "adc_hi": float(adc_hi),
        "normalize_for_png": bool(normalize),
        "saved_npy_is_raw": True,
        "plane": int(plane),
        "reason": str(tag),
        "diagnostics": diagnostics,
    }
    meta_path = out_prefix.with_name(out_prefix.name + f"_meta_{layer_name}.txt")
    meta_path.write_text(json.dumps(meta, indent=2))
    print("[ok]", meta_path)

    if save_npy:
        np.save(out_prefix.with_name(out_prefix.name + "_input.npy"), original_img)
        np.save(out_prefix.with_name(out_prefix.name + f"_signal_map_{layer_name}.npy"), sig_map_raw)
        np.save(out_prefix.with_name(out_prefix.name + f"_background_map_{layer_name}.npy"), bkg_map_raw)


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

    p.add_argument("--from-csv", default=None, help="CSV with columns: root_file, entry_number, out_dir, tag, n_pixels (plus optional weight_file, model, layer_name)")
    p.add_argument("--larcv-base", default="/data", help="Base path inside container for ROOTs (default /data)")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--tag", default="")

    p.add_argument("--adc-lo", type=float, default=10.0)
    p.add_argument("--adc-hi", type=float, default=500.0)
    p.add_argument("--normalize", action="store_true")

    p.add_argument("--plane", type=int, default=0)
    p.add_argument("--save-npy", action="store_true")
    p.add_argument("--gpuid", default="0")
    p.add_argument("--n-pixels", type=int, default=None)

    p.add_argument("--layer-name", default=None, help="Layer to hook (overrides --layer-preset)")
    p.add_argument("--layer-preset", default="mid32", choices=["mid128", "mid64", "mid32", "final"])

    # Diagnostics
    p.add_argument("--active-threshold", type=float, default=0.0)
    p.add_argument("--topk-frac", type=float, default=0.01)
    p.add_argument("--diag-curves", action="store_true")
    p.add_argument("--curve-steps", type=int, default=11)
    p.add_argument("--curve-seed", type=int, default=123)

    # Score-CAM runtime controls
    p.add_argument("--scorecam-max-channels", type=int, default=64)
    p.add_argument("--scorecam-batch", type=int, default=16)
    p.add_argument("--scorecam-channel-mode", default="topk", choices=["topk", "first"])
    p.add_argument("--scorecam-no-relu-acts", action="store_true")

    # Upsampling choice for masks + final CAM (nearest often looks cleaner for sparse tracks)
    p.add_argument("--cam-upsample", default="nearest", choices=["nearest", "bilinear"])
    return p


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

            layer_name = None
            if "layer_name" in df.columns and isinstance(row.get("layer_name", None), str) and row["layer_name"].strip():
                layer_name = str(row["layer_name"]).strip()

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
                adc_lo=args.adc_lo,
                adc_hi=args.adc_hi,
                normalize=args.normalize,
                plane=args.plane,
                save_npy=args.save_npy,
                layer_name=layer_name or args.layer_name,
                layer_preset=args.layer_preset,
                active_threshold=args.active_threshold,
                topk_frac=args.topk_frac,
                diag_curves=args.diag_curves,
                curve_steps=args.curve_steps,
                curve_seed=args.curve_seed,
                scorecam_max_channels=args.scorecam_max_channels,
                scorecam_batch=args.scorecam_batch,
                scorecam_channel_mode=args.scorecam_channel_mode,
                scorecam_use_relu_acts=(not args.scorecam_no_relu_acts),
                cam_upsample=args.cam_upsample,
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
            adc_lo=args.adc_lo,
            adc_hi=args.adc_hi,
            normalize=args.normalize,
            plane=args.plane,
            save_npy=args.save_npy,
            layer_name=args.layer_name,
            layer_preset=args.layer_preset,
            active_threshold=args.active_threshold,
            topk_frac=args.topk_frac,
            diag_curves=args.diag_curves,
            curve_steps=args.curve_steps,
            curve_seed=args.curve_seed,
            scorecam_max_channels=args.scorecam_max_channels,
            scorecam_batch=args.scorecam_batch,
            scorecam_channel_mode=args.scorecam_channel_mode,
            scorecam_use_relu_acts=(not args.scorecam_no_relu_acts),
            cam_upsample=args.cam_upsample,
            device=device,
        )


if __name__ == "__main__":
    main()
