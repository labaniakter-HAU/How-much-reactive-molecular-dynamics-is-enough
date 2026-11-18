# file: experiments/round4_synth_score.py
"""Round 4, synthetic part (design section 15): C4 and C6 scored on C5's round-3 confirmation networks (seeds
21000-23999, the same 3000 networks as C5), with round 2's bars. Written before the networks were rerun.

  python -I experiments/round2_synthetic.py --seeds 21000 21250 --target 3100 --out results/round4/synth_lumped_T3100.jsonl
  ... (12 blocks: lumped 21000, possible 22000, impossible 23000; 250 seeds per target 3100, 3600, 4500, 5000)
  python -I experiments/round4_synth_score.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from round2_score import h5, h7  # noqa: E402

OUT, R3 = ROOT / "results" / "round4", ROOT / "results" / "round3"
BLOCKS, TARGETS = {"lumped": 21000, "possible": 22000, "impossible": 23000}, (3100, 3600, 4500, 5000)
CODE = [ROOT / "src" / f for f in ("design.py", "fit.py", "integrate.py", "kinetics.py", "synthetic.py", "data.py")] + \
       [ROOT / "experiments" / f for f in ("round2_synthetic.py", "v0b_reduction.py", "v0c_synthetic.py", "round2_score.py",
                                           "round4_synth_score.py")]


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def main():
    frozen = json.loads((OUT / "FROZEN.json").read_text())
    assert frozen["synthetic_code_sha256"] == sha(CODE), "code changed since the freeze"
    assert not (OUT / "synth_score.json").exists(), "scored once"
    r2 = json.loads((ROOT / "results" / "round2" / "FROZEN.json").read_text())
    rows = {}
    for kind, s0 in BLOCKS.items():
        for i, T in enumerate(TARGETS):
            new = [json.loads(x) for x in (OUT / f"synth_{kind}_T{T}.jsonl").read_text().splitlines()]
            old = {r["seed"]: r for r in map(json.loads, (R3 / f"confirm_{kind}_T{T}.jsonl").read_text().splitlines())}
            assert sorted(r["seed"] for r in new) == list(range(s0 + 250 * i, s0 + 250 * (i + 1))), f"{kind} {T}: seeds"
            for r in new:  # the same networks as C5's confirmation: its values and C4's z must match round 3's
                o = old[r["seed"]]
                assert np.allclose(r["z_round2re"], o["z_round2re"], rtol=0, atol=1e-12) and np.isclose(r["full"], o["full"], rtol=0, atol=1e-12)
                assert all(np.isclose(r["c5"][f][k], o["c5"][f][k], rtol=0, atol=1e-12)
                           for f in r["c5"] for k in ("designed", "hot_only", "uniform", "median_random")), f"seed {r['seed']}"
            rows.setdefault(kind, []).extend(new)
    res = {f"C4 {kind}": h5(np.concatenate([r["z_round2re"] for r in rs]), r2["H5"]) for kind, rs in rows.items()}

    def c6(rs):
        cs = [c for r in rs for c in r["c6"]]
        return h7([c["rms_re"] > r2["label_rms"] for c in cs], [c["single"] > 0.5 or c["err_re"] > r2["flag_err"] for c in cs], r2["H7"])
    res["C6 all"] = c6([r for rs in rows.values() for r in rs])
    res.update({f"C6 {kind} (descriptive)": c6(rs) for kind, rs in rows.items()})
    res["networks"] = {kind: len(rs) for kind, rs in rows.items()}
    (OUT / "synth_score.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
