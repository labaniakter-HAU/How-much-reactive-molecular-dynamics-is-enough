# file: experiments/round5_score_syn.py
"""Round 5, C6 synthetic confirmation scorer (design section 16): H7' on each kind separately (lumped, possible,
impossible), as fixed in results/round5/FROZEN_c6.json. Written 2026-11-11 before any confirmation block was read;
its hash is in results/round5/FROZEN_c6_scorer.json. round5_confirm.py --score also reports the pooled value.

  python -I -X utf8 experiments/round5_score_syn.py      # once, after all 12 blocks
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import foresight  # noqa: E402
from round2_score import h7  # noqa: E402

OUT = ROOT / "results" / "round5"
KINDS, TARGETS = {"lumped": 40000, "possible": 41000, "impossible": 42000}, (3100, 3600, 4500, 5000)


def main():
    assert json.loads((OUT / "FROZEN_c6_scorer.json").read_text())["sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert not (OUT / "score_syn.json").exists(), "scored once"
    bar = json.loads((OUT / "FROZEN_c6.json").read_text())["bars"]["C6"]
    out, allrows = {}, []
    for kind, s0 in KINDS.items():
        rows = []
        for i, T in enumerate(TARGETS):
            nets = [json.loads(x) for x in (OUT / f"confirm_flag_syn_{kind}_T{T}.jsonl").read_text().splitlines()]
            assert sorted(r["seed"] for r in nets) == list(range(s0 + 250 * i, s0 + 250 * (i + 1))), f"{kind} {T}: seeds"
            assert all(r["target"] == T for r in nets)
            ok_kind = {"lumped": {"lumped"}, "possible": {"possible"}, "impossible": {"curved", "overdispersed", "both"}}[kind]
            assert all(r["kind"] in ok_kind for r in nets), f"{kind} {T}: network kinds {set(r['kind'] for r in nets)}"
            rows += [c for r in nets for c in r["campaigns"]]
        out[kind] = h7([c["rms"] > foresight.CUT for c in rows], [foresight.flagged(c["P"]) for c in rows], bar)
        out[kind]["frozen flag (descriptive)"] = h7([c["rms"] > foresight.CUT for c in rows], [c["old"] for c in rows], bar)
        allrows += rows
    out["pooled (descriptive)"] = h7([c["rms"] > foresight.CUT for c in allrows], [foresight.flagged(c["P"]) for c in allrows], bar)
    (OUT / "score_syn.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
