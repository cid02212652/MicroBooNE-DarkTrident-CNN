#!/usr/bin/env python3
"""Post-process XAI outputs (meta + optional npy) into analysis-friendly tables.

This script is intentionally *lossless with respect to the meta file*: it can
export 3 complementary CSVs:

1) events table (1 row per meta file)
2) masks table (1 row per meta file per {target, selection_mode, frac})
3) curves table (1 row per meta file per {curve_kind, target, frac})

Typical use:
  python xai_post_metrics_v2.py --out-dir /path/to/out \
      --events-csv events.csv --masks-csv masks.csv --curves-csv curves.csv

If you only want a single file:
  python xai_post_metrics_v2.py --out-dir /path/to/out --events-csv summary.csv

Notes
-----
- Meta files are expected to be JSON stored in *.txt (as produced by the CLIs).
- The "active" mask in meta-based diagnostics is whatever your run used
  (active_threshold, plane, adc clamp etc.).
- This script does *not* require loading npy files; it only reads meta.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _find_meta_files(out_dir: Path) -> List[Path]:
    pats = ["*_meta*.txt", "*_meta*.json"]
    files: List[Path] = []
    for pat in pats:
        files.extend(out_dir.rglob(pat))
    return sorted(set(files))


def _safe_get(d: Any, *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


    # try to infer from input_file or reason
    candidates = [str(meta.get("input_file") or ""), str(meta.get("reason") or "")]
    patts = [
        r"mass[_-]?(\d+(?:\.\d+)?)",      # mass500, mass_500
        r"(\d+(?:\.\d+)?)\s*(?:mev|gev)", # 500MeV
    ]
    for s in candidates:
        low = s.lower()
        for p in patts:
            m = re.search(p, low)
            if m:
                return m.group(0)
    return None


def _flatten_events(meta: Dict[str, Any]) -> Dict[str, Any]:
    diag = meta.get("diagnostics") if isinstance(meta.get("diagnostics"), dict) else {}

    out: Dict[str, Any] = {
        # identity
        "method": meta.get("method"),
        "model": meta.get("model"),
        "layer_name": meta.get("layer_name"),
        "plane": meta.get("plane"),
        "entry_number": meta.get("entry_number"),
        "reason": meta.get("reason"),
        "mass_token": meta.get("mass_token"),
        # provenance
        "weight_file": meta.get("weight_file"),
        "input_file": meta.get("input_file"),
        # run config
        "n_pixels": meta.get("n_pixels"),
        "adc_lo": meta.get("adc_lo"),
        "adc_hi": meta.get("adc_hi"),
        "occlusion_size": meta.get("occlusion_size"),
        "stride": meta.get("stride"),
        "normalize_for_png": meta.get("normalize_for_png"),
        "saved_npy_is_raw": meta.get("saved_npy_is_raw"),
        # base scores
        "base_signal_score": meta.get("base_signal_score"),
        "base_background_score": meta.get("base_background_score"),
        # basic diagnostics
        "active_threshold": _safe_get(diag, "active_threshold"),
        "active_frac": _safe_get(diag, "active_frac"),
        "topk_frac": _safe_get(diag, "topk_frac"),
        "overlap_topk_signal": _safe_get(diag, "overlap_topk_signal"),
        "overlap_topk_background": _safe_get(diag, "overlap_topk_background"),
        "has_mask_diagnostics": isinstance(_safe_get(diag, "mask_diagnostics"), dict),
        "has_attr_adc_correlation": isinstance(_safe_get(diag, "attr_adc_correlation"), dict),
        "has_curves": isinstance(_safe_get(diag, "curves"), dict),
        "has_pixel_need": isinstance(_safe_get(diag, "curves_pixel_need"), dict),
    }

    # occlusion specifics (if present)
    out["occlusion_attribution"] = _safe_get(diag, "occlusion_attribution")
    out["occlusion_patch_halfwidth"] = _safe_get(diag, "occlusion_patch_halfwidth")

    # mask_diagnostics summary scalars (not the whole fraction grid)
    md = _safe_get(diag, "mask_diagnostics")
    if isinstance(md, dict):
        for target in ["signal", "background"]:
            td = md.get(target) if isinstance(md.get(target), dict) else {}
            out[f"{target}_n_total"] = td.get("n_total")
            out[f"{target}_n_active"] = td.get("n_active")
            out[f"{target}_use_abs_rank"] = td.get("use_abs_rank")
            out[f"{target}_active_abs_mass_frac"] = _safe_get(td, "active_abs_mass_frac")
            # distribution stats (on active)
            stats = _safe_get(td, "attr_stats_on_active")
            if isinstance(stats, dict):
                for k in ["min", "max", "mean", "median", "q90", "q95", "q99", "q999"]:
                    out[f"{target}_attr_{k}_on_active"] = stats.get(k)
            # sign stats (on active)
            sstats = _safe_get(td, "sign_stats_on_active")
            if isinstance(sstats, dict):
                for k in ["active_pos_frac", "active_neg_frac", "active_zero_frac"]:
                    out[f"{target}_{k}"] = sstats.get(k)

    # attr-adc correlation
    corr = _safe_get(diag, "attr_adc_correlation")
    if isinstance(corr, dict):
        for target in ["signal", "background"]:
            cd = corr.get(target) if isinstance(corr.get(target), dict) else {}
            for k in ["pearson_r", "pearson_p", "spearman_r", "spearman_p", "n"]:
                out[f"{target}_adc_{k}"] = cd.get(k)

    # curves + AUC + advantage over random
    curves = _safe_get(diag, "curves")
    if isinstance(curves, dict):
        auc = curves.get("auc") if isinstance(curves.get("auc"), dict) else {}
        # raw AUCs
        for k, v in auc.items():
            out[f"auc_{k}"] = v
        # advantages (positive means better than random)
        # insertion: higher is better
        ins_sig = _as_float(auc.get("insertion_signal"))
        ins_sig_r = _as_float(auc.get("random_insertion_signal"))
        if ins_sig is not None and ins_sig_r is not None:
            out["auc_insertion_signal_adv"] = ins_sig - ins_sig_r
        ins_bkg = _as_float(auc.get("insertion_background"))
        ins_bkg_r = _as_float(auc.get("random_insertion_background"))
        if ins_bkg is not None and ins_bkg_r is not None:
            out["auc_insertion_background_adv"] = ins_bkg - ins_bkg_r
        # deletion: lower is better, so advantage can be (random - method)
        del_sig = _as_float(auc.get("deletion_signal"))
        del_sig_r = _as_float(auc.get("random_deletion_signal"))
        if del_sig is not None and del_sig_r is not None:
            out["auc_deletion_signal_adv"] = del_sig_r - del_sig
        del_bkg = _as_float(auc.get("deletion_background"))
        del_bkg_r = _as_float(auc.get("random_deletion_background"))
        if del_bkg is not None and del_bkg_r is not None:
            out["auc_deletion_background_adv"] = del_bkg_r - del_bkg

    # pixel need
    cpn = _safe_get(diag, "curves_pixel_need")
    if isinstance(cpn, dict):
        # Store all thresholds for both signal/background
        for target in ["signal", "background"]:
            reach = _safe_get(cpn, f"insertion_{target}_frac_to_reach")
            drop = _safe_get(cpn, f"deletion_{target}_frac_to_drop")
            if isinstance(reach, dict):
                for k, v in reach.items():
                    out[f"ins_{target}_frac_to_{k}"] = v
            if isinstance(drop, dict):
                for k, v in drop.items():
                    out[f"del_{target}_frac_to_{k}"] = v

            # Also convert to number of pixels if n_total known
            n_total = _as_float(out.get(f"{target}_n_total"))
            if n_total is not None:
                if isinstance(reach, dict):
                    for k, v in reach.items():
                        fv = _as_float(v)
                        if fv is not None:
                            out[f"ins_{target}_pixels_to_{k}"] = fv * n_total
                if isinstance(drop, dict):
                    for k, v in drop.items():
                        fv = _as_float(v)
                        if fv is not None:
                            out[f"del_{target}_pixels_to_{k}"] = fv * n_total

    # what attribution used for curves
    out["curves_attr_used"] = _safe_get(diag, "curves_attr_used")

    return out


def _iter_mask_rows(meta: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    diag = meta.get("diagnostics") if isinstance(meta.get("diagnostics"), dict) else {}
    md = diag.get("mask_diagnostics") if isinstance(diag.get("mask_diagnostics"), dict) else None
    if not isinstance(md, dict):
        return

    base = {
        "method": meta.get("method"),
        "model": meta.get("model"),
        "layer_name": meta.get("layer_name"),
        "plane": meta.get("plane"),
        "entry_number": meta.get("entry_number"),
        "reason": meta.get("reason"),
        "mass_token": meta.get("mass_token"),
        "input_file": meta.get("input_file"),
        "weight_file": meta.get("weight_file"),
    }

    for target in ["signal", "background"]:
        td = md.get(target) if isinstance(md.get(target), dict) else {}
        for mode in ["global_topk", "active_only_topk"]:
            rows = td.get(mode)
            if not isinstance(rows, list):
                continue
            for r in rows:
                if not isinstance(r, dict):
                    continue
                out = dict(base)
                out.update({
                    "target": target,
                    "selection_mode": mode,
                    "frac": r.get("frac"),
                    "n_selected": r.get("n_selected"),
                    "precision_on_active": r.get("precision_on_active"),
                    "recall_active_flagged": r.get("recall_active_flagged"),
                    "iou_with_active": r.get("iou_with_active"),
                    "frag_n_components": r.get("frag_n_components"),
                    "frag_largest_component_frac": r.get("frag_largest_component_frac"),
                    "n_total": td.get("n_total"),
                    "n_active": td.get("n_active"),
                    "active_frac": td.get("active_frac"),
                    "use_abs_rank": td.get("use_abs_rank"),
                })
                yield out


def _iter_curve_rows(meta: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    diag = meta.get("diagnostics") if isinstance(meta.get("diagnostics"), dict) else {}
    curves = diag.get("curves") if isinstance(diag.get("curves"), dict) else None
    if not isinstance(curves, dict):
        return

    fracs = curves.get("fractions")
    if not isinstance(fracs, list) or not fracs:
        return

    base = {
        "method": meta.get("method"),
        "model": meta.get("model"),
        "layer_name": meta.get("layer_name"),
        "plane": meta.get("plane"),
        "entry_number": meta.get("entry_number"),
        "reason": meta.get("reason"),
        "mass_token": meta.get("mass_token"),
        "input_file": meta.get("input_file"),
        "weight_file": meta.get("weight_file"),
        "curves_attr_used": _safe_get(diag, "curves_attr_used"),
    }

    for kind in ["deletion", "insertion", "random_deletion", "random_insertion"]:
        kd = curves.get(kind)
        if not isinstance(kd, dict):
            continue
        for target in ["signal", "background"]:
            ys = kd.get(target)
            if not isinstance(ys, list) or len(ys) != len(fracs):
                continue
            for f, y in zip(fracs, ys):
                out = dict(base)
                out.update({
                    "curve_kind": kind,
                    "target": target,
                    "frac": f,
                    "score": y,
                })
                yield out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--events-csv", default=None, help="Write 1 row per meta file")
    ap.add_argument("--masks-csv", default=None, help="Write long table for mask_diagnostics")
    ap.add_argument("--curves-csv", default=None, help="Write long table for curves")

    ap.add_argument(
        "--print-head",
        type=int,
        default=0,
        help="If >0, print first N rows of events table to stdout",
    )
    ap.add_argument(
        "--group-summary",
        action="store_true",
        help="Print grouped means/stderr for a few core metrics from events table",
    )
    ap.add_argument(
        "--group-by",
        default="mass_token,method,model",
        help="Comma-separated group keys for --group-summary",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    metas = _find_meta_files(out_dir)
    if not metas:
        raise SystemExit(f"No meta files found under {out_dir}")

    event_rows: List[Dict[str, Any]] = []
    mask_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    for mp in metas:
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            continue

        ev = _flatten_events(meta)
        ev["meta_path"] = str(mp)
        event_rows.append(ev)

        if args.masks_csv:
            for r in _iter_mask_rows(meta):
                r["meta_path"] = str(mp)
                mask_rows.append(r)

        if args.curves_csv:
            for r in _iter_curve_rows(meta):
                r["meta_path"] = str(mp)
                curve_rows.append(r)

    df_events = pd.DataFrame(event_rows)

    if args.events_csv:
        Path(args.events_csv).parent.mkdir(parents=True, exist_ok=True)
        df_events.to_csv(args.events_csv, index=False)
        print(f"[ok] wrote events: {args.events_csv} ({len(df_events)} rows)")

    if args.masks_csv:
        df_masks = pd.DataFrame(mask_rows)
        Path(args.masks_csv).parent.mkdir(parents=True, exist_ok=True)
        df_masks.to_csv(args.masks_csv, index=False)
        print(f"[ok] wrote masks:  {args.masks_csv} ({len(df_masks)} rows)")

    if args.curves_csv:
        df_curves = pd.DataFrame(curve_rows)
        Path(args.curves_csv).parent.mkdir(parents=True, exist_ok=True)
        df_curves.to_csv(args.curves_csv, index=False)
        print(f"[ok] wrote curves: {args.curves_csv} ({len(df_curves)} rows)")

    if args.print_head and args.print_head > 0:
        print(df_events.head(int(args.print_head)).to_string(index=False))

    if args.group_summary:
        keys = [k.strip() for k in str(args.group_by).split(",") if k.strip()]
        keys = [k for k in keys if k in df_events.columns]
        if not keys:
            print("[warn] no valid group-by keys found")
            return

        # Choose a few metrics that tend to exist
        candidates = [
            "auc_insertion_signal_adv",
            "auc_deletion_signal_adv",
            "ins_signal_frac_to_90%",
            "del_signal_frac_to_50%",
            "signal_active_abs_mass_frac",
            "overlap_topk_signal",
            "active_frac",
        ]
        metrics = [c for c in candidates if c in df_events.columns]
        if not metrics:
            print("[warn] no metrics found for group summary")
            return

        g = df_events.groupby(keys)[metrics].agg(["count", "mean", "std"]).reset_index()
        print("\n=== Group summary ===")
        print(g.to_string(index=False))


if __name__ == "__main__":
    main()