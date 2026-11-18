# file: experiments/round6_confirm.py
"""Round 6 confirmation (design section 17): C4 with the variance variant frozen in results/round6/FROZEN.json, once on
  MEA: every non-full campaign of round 4's 5 MEA folds (C5's campaigns), z as round 4 (observation variance 1/n* + s2/4)
  synthetic: fresh seeds 43000-45999 (lumped 43000-43999, possible 44000-44999, impossible 45000-45999; 250 per target
  in the order 3100, 3600, 4500, 5000), the full campaign of each network, z against the true ln k (as rounds 2 and 4)
Bars: H5' (>= 85 % of |z| <= 2, median |z| 0.5-1.0) on MEA and on each synthetic kind.

  python -I -X utf8 experiments/round6_confirm.py --mea
  python -I -X utf8 experiments/round6_confirm.py --synthetic 43000 43250 --target 3100 --workers 11
  python -I -X utf8 experiments/round6_confirm.py --score       # once
"""
import argparse
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

OUT = ROOT / "results" / "round6"
CODE = [ROOT / "src" / f for f in ("design.py", "species.py", "lab.py", "fit.py", "kinetics.py", "integrate.py", "data.py", "synthetic.py")] + \
       [ROOT / "experiments" / f for f in ("round6_confirm.py", "round6_dev_c4.py", "round4_lab.py", "c5_lab.py", "round5_flag.py",
                                           "round2_synthetic.py", "v0b_reduction.py", "v0c_synthetic.py")]
KINDS, TARGETS = {"lumped": 43000, "possible": 44000, "impossible": 45000}, (3100, 3600, 4500, 5000)


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def check():
    import c5_lab
    frozen = json.loads((OUT / "FROZEN.json").read_text())  # raises if round 6 is not frozen
    assert frozen["code_sha256"] == sha(CODE), "code changed since the freeze"
    assert frozen["data_sha256"] == sha([c5_lab.MANIFEST]) and c5_lab.MANIFEST.read_text().splitlines()[2:] == c5_lab.manifest(), "data changed"
    return frozen


def mea(variant):
    import round4_lab as r4
    from round6_dev_c4 import variances
    z = []
    for T_star in r4.TEMPS:
        T_of, data, S, O, groups, menu, A, pilot, train, test = r4.setup("MEA", T_star)
        camps = r4.campaigns("MEA", T_star, T_of, data, menu, A, pilot, test)
        obs, w = r4.observed(test, data, T_of, A)
        keep = w >= 10
        for name, camp in camps.items():
            if name == "full":
                continue
            mean, s2, sds = variances(camp, data, T_of, A, T_star)
            z += ((mean - obs) / np.sqrt(sds[variant] ** 2 + 1.0 / np.maximum(w, 1.0) + s2 / len(test)))[keep].round(5).tolist()
    return z


def confirm_one(args):
    """Confirmation seeds 43000-45999 only, and only once round 6 is frozen."""
    seed, T_star = args
    assert 43000 <= seed < 46000 and (OUT / "FROZEN.json").exists()
    from round6_dev_c4 import synthetic_core
    return synthetic_core(seed, T_star)


def h5(z):
    a = np.abs(np.asarray(z, float))
    a = a[np.isfinite(a)]
    cov, med = float((a <= 2).mean()), float(np.median(a))
    return {"n": int(a.size), "coverage": cov, "median_abs_z": med, "pass": bool(cov >= 0.85 and 0.5 <= med <= 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mea", action="store_true")
    ap.add_argument("--synthetic", type=int, nargs=2)
    ap.add_argument("--target", type=float)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    frozen = check()
    variant = frozen["variant"]
    if a.mea:
        path = OUT / "confirm_mea.json"
        assert not path.exists(), "run once"
        path.write_text(json.dumps({"variant": variant, "z": mea(variant)}))
        print("MEA done")
    if a.synthetic:
        seeds = list(range(*a.synthetic))
        assert all(43000 <= s < 46000 for s in seeds)
        kind = next(k for k, s0 in KINDS.items() if s0 <= seeds[0] < s0 + 1000)
        path = OUT / f"confirm_syn_{kind}_T{a.target:g}.jsonl"
        assert not path.exists() or len(path.read_text().splitlines()) < len(seeds), "run once"
        with Pool(a.workers) as pool, open(path, "w") as f:
            for r in pool.imap_unordered(confirm_one, [(s, a.target) for s in seeds]):
                f.write(json.dumps({"seed": r["seed"], "kind": r["kind"], "target": r["target"], "z": r["z"][variant]}) + "\n")
                f.flush()
        print(f"{kind} {a.target:g}: {len(seeds)} networks")
    if a.score:
        assert not (OUT / "score.json").exists(), "scored once"
        out = {"variant": variant, "MEA": h5(json.loads((OUT / "confirm_mea.json").read_text())["z"])}
        for kind, s0 in KINDS.items():
            z = []
            for i, T in enumerate(TARGETS):
                rows = [json.loads(x) for x in (OUT / f"confirm_syn_{kind}_T{T}.jsonl").read_text().splitlines()]
                assert sorted(r["seed"] for r in rows) == list(range(s0 + 250 * i, s0 + 250 * (i + 1))), f"{kind} {T}: seeds"
                ok_kind = {"lumped": {"lumped"}, "possible": {"possible"}, "impossible": {"curved", "overdispersed", "both"}}[kind]
                assert all(r["kind"] in ok_kind and r["target"] == T for r in rows), f"{kind} {T}: kinds or targets"
                z += [x for r in rows for x in r["z"]]
            out[f"synthetic {kind}"] = h5(z)
        (OUT / "score.json").write_text(json.dumps(out, indent=1))
        for k, v in out.items():
            print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
