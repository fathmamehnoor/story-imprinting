"""Cosines between story directions, and the shared and differing parts of an hb_dc and an hc_db direction.

  python steering/combine_directions.py OUT_DIR base=results/steering/directions/base.pt hb=PATH_hb_dc.pt hc=PATH_hc_db.pt [more=...]

Prints the cosine between every pair. If "hb" and "hc" are given, writes
  OUT_DIR/shared.pt        normalised sum of the two unit directions
  OUT_DIR/hb_minus_hc.pt   normalised difference (hb minus hc)
each with sd_gate = the mean of the two directions' sd_gate, so steering strengths are in comparable units.
"""
import sys
from pathlib import Path
import torch

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
d = {k: torch.load(p, map_location="cpu", weights_only=True) for k, p in (s.split("=", 1) for s in sys.argv[2:])}
for k, v in d.items():
    print(f"{k:8s} layer {v['layer']}  gate d' {v['d_prime_gate']:.2f}  AUC {v['auc_gate']:.3f}  one story SD = {v['sd_gate']:.2f} residual units")
ks = list(d)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        print(f"cosine {ks[i]} vs {ks[j]}: {float(d[ks[i]]['unit'] @ d[ks[j]]['unit']):.3f}")
if "hb" in d and "hc" in d:
    assert d["hb"]["layer"] == d["hc"]["layer"]
    sd = (d["hb"]["sd_gate"] + d["hc"]["sd_gate"]) / 2
    for name, v in (("shared", d["hb"]["unit"] + d["hc"]["unit"]), ("hb_minus_hc", d["hb"]["unit"] - d["hc"]["unit"])):
        print(f"{name}: norm before normalising {float(v.norm()):.3f}")
        torch.save({"unit": v / v.norm(), "layer": d["hb"]["layer"], "sd_gate": sd}, out / f"{name}.pt")
