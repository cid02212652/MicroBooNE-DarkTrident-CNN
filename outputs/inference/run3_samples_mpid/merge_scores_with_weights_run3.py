#!/usr/bin/env python3
import numpy as np
import pandas as pd

KEYS = ["run_number", "subrun_number", "event_number"]

PAIRS = [
    ("run3_dirt_larcv_cropped_scores.csv",
     "run3_dirt_CNN.csv",
     "run3_dirt_merged_with_weights.csv"),

    ("run3_nu_overlay_larcv_cropped_scores.csv",
     "run3_nu_overlay_CNN.csv",
     "run3_nu_overlay_merged_with_weights.csv"),
]

MERGE_HOW = "inner"  # change to "left" if you want to keep all score rows

def force_int_keys(df, keys):
    for k in keys:
        # robust cast in case they were read as floats/strings
        df[k] = pd.to_numeric(df[k], errors="raise").astype("int64")
    return df

def clean_weight(arr):
    arr = arr.astype(float, copy=False)
    bad = (~np.isfinite(arr)) | (arr <= 0) | (arr > 50)
    arr[bad] = 1.0
    return arr

for scores_file, weights_file, out_file in PAIRS:
    print("\nMerging:")
    print("  scores :", scores_file)
    print("  weights:", weights_file)

    # --- read scores
    scores = pd.read_csv(scores_file, usecols=KEYS + ["signal_score"])
    scores = force_int_keys(scores, KEYS)

    # If duplicate score keys exist, keep the first (should usually be unique)
    dup_scores = scores.duplicated(KEYS).sum()
    if dup_scores:
        print(f"  [warn] duplicate score keys: {dup_scores} (keeping first per key)")
        scores = scores.drop_duplicates(KEYS, keep="first")

    # --- read weights inputs
    wdf = pd.read_csv(weights_file, usecols=KEYS + ["ppfx_cv_good", "spine_tune_good"])
    wdf = force_int_keys(wdf, KEYS)

    # compute product weight
    w = (wdf["ppfx_cv_good"].to_numpy(dtype=float) *
         wdf["spine_tune_good"].to_numpy(dtype=float))

    # clean weights: inf/nan/<=0/>50 -> 1
    w = clean_weight(w)

    weights = wdf[KEYS].copy()
    weights["weight"] = w

    # If duplicate weight keys exist, average their weights (avoids cartesian duplication on merge)
    dup_w = weights.duplicated(KEYS).sum()
    if dup_w:
        print(f"  [warn] duplicate weight keys: {dup_w} (averaging weight per key)")
        weights = weights.groupby(KEYS, as_index=False)["weight"].mean()

    # --- merge
    merged = scores.merge(weights, on=KEYS, how=MERGE_HOW)

    # If left-merge, any missing weights -> 1 (for inner-merge this is usually unnecessary)
    if MERGE_HOW != "inner":
        merged["weight"] = merged["weight"].fillna(1.0)

    print(f"  scores rows:  {len(scores)}")
    print(f"  weights rows: {len(weights)}")
    print(f"  merged rows:  {len(merged)}")
    print(f"  mean weight:  {merged['weight'].mean():.6f}")

    merged.to_csv(out_file, index=False)
    print("  ->", out_file)
