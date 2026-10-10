"""Steering direction file for the Axis (laptop), in steering/build_direction.py's format, for steering/steer_logprob.py
and axis/steer_sample.py.

  python -m axis.make_direction            # the measure axis.pt names (`use`)
  python -m axis.make_direction --measure turn

unit     the Axis at layer 36, unit length, pointing towards the default Assistant (as built)
sd_gate  the size of one strength unit: how far the dismissive system prompt itself moves the end_of_turn state along
         the Axis (mean over the 100 probe conversations, base model, fixed history, after the prohibition). So
         strength -1 moves the state away from the Assistant by as much as the dismissive prompt does, +1 the
         same distance towards it.
Writes runs/axis/axis_dir.pt {unit, layer, sd_gate, raw_norm, measure, dismissive_shift} and axis_dir.txt.
"""
from __future__ import annotations

import argparse

import numpy as np

from .common import LAYER, OUT
from .select_pairs import dismissive_shift


def main(argv=None) -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", choices=["reply", "turn"], default=None)
    ap.add_argument("--out", default=str(OUT / "axis_dir.pt"))
    args = ap.parse_args(argv)
    ax = torch.load(OUT / "axis.pt", map_location="cpu", weights_only=False)
    m = args.measure or ax["use"]
    li = ax["layers"].index(LAYER)
    raw = ax["axis"][m][li].float()
    u = raw / raw.norm()
    shift = dismissive_shift(u.numpy().astype(np.float64), li)
    torch.save({"unit": u, "layer": LAYER, "sd_gate": abs(shift), "raw_norm": float(raw.norm()), "measure": m,
                "dismissive_shift": shift}, args.out)
    text = (f"Axis ({m}) at layer {LAYER}{'' if m == 'reply' else ' -- a new measure, not the paper Axis'}: unit points "
            f"towards the default Assistant. The dismissive prompt moves the end_of_turn state {shift:+.3f} along it, so "
            f"one strength unit = {abs(shift):.3f} residual units (the story direction's unit was 4.89).")
    if shift >= 0:
        text += " WARNING: the dismissive prompt moves the state towards the Assistant here."
    open(str(args.out).replace(".pt", ".txt"), "w").write(text + "\n")
    print(text + f"\n-> {args.out}")


if __name__ == "__main__":
    main()
