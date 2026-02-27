import glob, csv, os
import uproot

def find_branch(keys, endswith):
    for k in keys:
        if k.endswith(endswith):
            return k
    return None

IN_DIR  = "/vols/sbn/uboone/darkTridents/data/larcv_files/run3_signal"
OUT_DIR = "/home/hep/an1522/dark_tridents_wspace/run3_signal"

os.makedirs(OUT_DIR, exist_ok=True)

for rootpath in sorted(glob.glob(IN_DIR + "/*.root")):
    rootfile = os.path.basename(rootpath)

    f = uproot.open(rootpath)
    fkeys = f.keys()
    if len(fkeys) == 0:
        print("SKIP (0 keys):", rootfile)
        continue

    tree = f[fkeys[0]]  # first tree
    keys = list(tree.keys())

    run_b    = find_branch(keys, "/_run")
    subrun_b = find_branch(keys, "/_subrun")
    event_b  = find_branch(keys, "/_event")

    if not (run_b and subrun_b and event_b):
        print("SKIP (no run/subrun/event):", rootfile)
        continue

    run = tree[run_b].array(library="np")
    sub = tree[subrun_b].array(library="np")
    evt = tree[event_b].array(library="np")

    out = os.path.join(OUT_DIR, rootfile[:-5] + ".csv")
    with open(out, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["run_number", "subrun_number", "event_number"])
        w.writerows(zip(run, sub, evt))

    print(rootfile, "->", out)
