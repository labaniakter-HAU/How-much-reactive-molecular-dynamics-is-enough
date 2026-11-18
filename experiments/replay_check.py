# file: experiments/replay_check.py
"""Data check for Plan 3: replaying each run's events from its first species frame must reproduce every
stored species frame exactly (species counts change only through events). Reads data only; no model.

  python -I experiments/replay_check.py data/processed/stanford
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402

from fullnet import Network, load_run  # noqa: E402


def main(tables):
    runs, bad = pd.read_csv(Path(tables) / "runs.csv"), 0
    for r in runs.itertuples():
        w, V, ev = load_run(Path(tables) / "species", r.run_id)
        net = Network.from_runs([(float(r.T_K), w, V, ev)])
        mismatch = net.replay_exposure(w, V, ev)[1]
        bad += mismatch > 0
        print(f"{r.run_id}: {len(ev)} events, {len(net.reactions)} reactions, max count mismatch {mismatch:g}", flush=True)
    print(f"**replay {'OK' if not bad else f'MISMATCH in {bad} runs'}**")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
