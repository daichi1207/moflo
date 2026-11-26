"""Compare an evaluation folder with reference/ (the paper's headline run).

Fails (exit 1) if any predicted pose differs by more than --tol metres / radians.
Success outcomes are reported but not enforced: individual policy rollouts can flip
between runs even when the predicted pose is identical.
"""
import argparse, glob, json, math, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--reference", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference"))
ap.add_argument("--tol", type=float, default=5e-5)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.results, "poc_*.json")))
if not files:
    sys.exit(f"no poc_*.json in {a.results}")
bad = n = 0
for p in files:
    ref = {(r["layout"], r["ep"]): r for r in json.load(open(os.path.join(a.reference, os.path.basename(p))))["records"]}
    recs = json.load(open(p))["records"]
    dxy = dth = agree = new = old = scored = 0
    for r in recs:
        o = ref.get((r["layout"], r["ep"]))
        if o is None:
            sys.exit(f"{p}: no reference for layout {r['layout']} episode {r['ep']}")
        d = math.dist(r["target"][:2], o["target"][:2])
        t = abs((r["target"][2] - o["target"][2] + math.pi) % (2 * math.pi) - math.pi)
        i = math.dist(r["init_pose"], o["init_pose"])
        bad += not (d <= a.tol and t <= a.tol and i <= a.tol)  # NaN counts as a mismatch
        n += 1
        dxy, dth = max(dxy, d), max(dth, t)
        if "success" in r:
            scored += 1; new += r["success"]; old += o["success"]; agree += r["success"] == o["success"]
    line = f"{os.path.basename(p)[4:-5]:17s} n={len(recs):3d}  max pose diff {dxy:.1e} m / {dth:.1e} rad"
    if scored:
        line += f"  success {new}/{scored} (reference {old}/{scored}, {agree} agree)"
    print(line)
print(f"poses: {n - bad}/{n} within {a.tol:g}")
sys.exit(bad > 0)
