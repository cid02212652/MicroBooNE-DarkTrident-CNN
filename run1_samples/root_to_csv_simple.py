import glob, csv
import uproot

def find_branch(keys, endswith):
    for k in keys:
        if k.endswith(endswith):
            return k
    return None

for rootfile in sorted(glob.glob("*.root")):
    f = uproot.open(rootfile)
    tree = f[f.keys()[0]]  # first (and usually only) tree

    keys = list(tree.keys())

    run_b    = find_branch(keys, "/_run")
    subrun_b = find_branch(keys, "/_subrun")
    event_b  = find_branch(keys, "/_event")

    if not (run_b and subrun_b and event_b):
        print("Could not find run/subrun/event in", rootfile)
        print("Example keys:", keys[:20])
        continue

    run  = tree[run_b].array(library="np")
    sub  = tree[subrun_b].array(library="np")
    evt  = tree[event_b].array(library="np")

    out = rootfile[:-5] + ".csv"  # replace .root with .csv
    with open(out, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["run_number", "subrun_number", "event_number"])
        w.writerows(zip(run, sub, evt))

    print(rootfile, "->", out)
