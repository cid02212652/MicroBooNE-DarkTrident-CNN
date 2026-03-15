#!/usr/bin/env python3
"""Compute pairwise IoU between *methods* for the same event using saved .npy maps.

Assumes each run saved:
  <prefix>_signal_map.npy
  <prefix>_input.npy (optional, not required)

Usage:
  python xai_pairwise_iou.py --out-dir /path/to/out --frac 0.01 --csv-out pairwise_iou.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def topk_mask(attr: np.ndarray, frac: float, use_abs: bool = True) -> np.ndarray:
    a = np.abs(attr) if use_abs else attr
    flat = a.reshape(-1)
    n = flat.size
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


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--frac', type=float, default=0.01)
    ap.add_argument('--csv-out', required=True)
    ap.add_argument('--use-abs', action='store_true', help='Rank by |attr| (default true)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    meta_files = sorted(out_dir.rglob('*_meta*.txt'))
    items: Dict[Tuple[str,int,int], List[Tuple[str, Path]]] = {}

    for mp in meta_files:
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            continue
        method = str(meta.get('method', ''))
        inp = str(meta.get('input_file', ''))
        entry = int(meta.get('entry_number', -1))
        plane = int(meta.get('plane', 0))

        # derive prefix: remove trailing _meta...txt
        stem = mp.name
        if '_meta' not in stem:
            continue
        prefix_name = stem.split('_meta')[0]
        prefix_path = mp.with_name(prefix_name)
        sig_npy = prefix_path.with_name(prefix_path.name + '_signal_map.npy')
        if not sig_npy.exists():
            continue

        key = (inp, entry, plane)
        items.setdefault(key, []).append((method, sig_npy))

    rows = []
    for (inp, entry, plane), lst in items.items():
        if len(lst) < 2:
            continue
        # load masks once
        masks = {}
        for method, sig_npy in lst:
            try:
                arr = np.load(sig_npy)
            except Exception:
                continue
            masks[method] = topk_mask(arr, args.frac, use_abs=True if args.use_abs else False)
        methods = sorted(masks.keys())
        for a, b in itertools.combinations(methods, 2):
            rows.append({
                'input_file': inp,
                'entry_number': entry,
                'plane': plane,
                'method_a': a,
                'method_b': b,
                f'iou@{args.frac}': iou(masks[a], masks[b]),
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out, index=False)
    print(f'[ok] wrote {args.csv_out} ({len(df)} pairwise rows)')


if __name__ == '__main__':
    main()
