"""Follow-up checks on the wording-vs-internals verdict (exploratory, after seeing it; not part of the preregistered test).

The preregistered count only uses the sign of each pair's behaviour difference. This adds:
  - per pair, the half-B behaviour difference T_B(a) - T_B(b) with a 95% bootstrap CI over the 50 conversations;
  - the contrast that stood out: the selected grumpy-but-helpful prompts ("negative") minus the selected
    busy-situation prompts ("implicit"), on half B, next to what the transfer rater predicted for them.
The CIs resample conversations, not prompts, so they say nothing about other prompts of the same kind.

  python -m persona_flip.wording_extras                  # reads runs/candidates/pairs.csv and runs/probe_cand_*
  python -m persona_flip.wording_extras --runs <folder>  # the same files under another folder
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from . import summarize_probe as sp
from .candidates import CATEGORY_OF
from .common import RUNS
from .stats import FT
from .wording_test import HALF_A, PROBE_NAME, read_scores

N_BOOT = 10_000


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS), help="folder holding candidates/ and the probe_cand files")
    args = ap.parse_args(argv)
    runs = Path(args.runs)
    sp.RUNS = runs
    P = {m: sp.per_prompt(m, PROBE_NAME) for m in ("base",) + FT}
    rows = lambda p: sp.condition_rows(P, (p, "fixed", "trigger/start"))
    r0 = rows("none")
    pairs = list(csv.DictReader(open(runs / "candidates" / "pairs.csv")))
    sel = [p for pr in pairs for p in (pr["a"], pr["b"])]
    half_b = sorted(q for q in r0 if q not in HALF_A)
    per = {p: np.array([-(rows(p)[q]["pooled"] - r0[q]["pooled"]) for q in half_b]) for p in sel}
    idx = np.random.default_rng(0).integers(0, len(half_b), size=(N_BOOT, len(half_b)))
    ci = lambda d: np.percentile(d[idx].mean(1), [2.5, 97.5])
    print("Exploratory, after seeing the verdict. Units: nats towards the dismissive character's animal, pooled.\n")
    print("Per pair: T_B(a) - T_B(b), 95% bootstrap CI over the 50 half-B conversations")
    for pr in pairs:
        d = per[pr["a"]] - per[pr["b"]]
        lo, hi = ci(d)
        call = "X" if lo > 0 else "wording" if hi < 0 else "too close to call"
        print(f"  {pr['a']:16s} vs {pr['b']:16s} {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]  {call}")
    scores = read_scores(runs / "wording" / "scores.csv")
    neg = [p for p in sel if CATEGORY_OF[p] == "negative"]
    imp = [p for p in sel if CATEGORY_OF[p] == "implicit"]
    if neg and imp:
        d = np.mean([per[p] for p in neg], 0) - np.mean([per[p] for p in imp], 0)
        lo, hi = ci(d)
        print(f"\nThe {len(neg)} selected 'negative' (grumpy but helpful) prompts minus the {len(imp)} 'implicit' (busy "
              f"situation) ones, half B: {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")
        print(f"  mean behaviour: negative {np.mean([per[p].mean() for p in neg]):+.2f}, implicit "
              f"{np.mean([per[p].mean() for p in imp]):+.2f}; the transfer rater predicted "
              f"{np.mean([scores[p]['gpt_transfer'] for p in neg]):.0f} vs {np.mean([scores[p]['gpt_transfer'] for p in imp]):.0f}")
    print("\nThe CIs resample conversations, not prompts; the contrast was chosen after seeing the results.")


if __name__ == "__main__":
    main()
