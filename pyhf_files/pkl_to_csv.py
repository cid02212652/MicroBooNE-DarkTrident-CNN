#!/usr/bin/env python3
import pandas as pd

FILES = [
    "run1_dirt_CNN.pkl",
    "run1_nu_overlay_CNN.pkl",
    "run3_dirt_CNN.pkl",
    "run3_nu_overlay_CNN.pkl",
]

for f in FILES:
    obj = pd.read_pickle(f)

    # Most likely case: it's already a DataFrame
    if isinstance(obj, pd.DataFrame):
        df = obj

    # If it's a Series, convert to a 1-col DataFrame
    elif isinstance(obj, pd.Series):
        df = obj.to_frame()

    # If it's something like dict/list, try to make a DataFrame out of it
    else:
        df = pd.DataFrame(obj)

    out = f.replace(".pkl", ".csv")
    df.to_csv(out, index=False)
    print(f"{f} -> {out}  (rows={len(df)}, cols={len(df.columns)})")
