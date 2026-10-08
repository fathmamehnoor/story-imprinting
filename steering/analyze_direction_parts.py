"""Steering slope of several directions in each fine-tune: the base model's, the fine-tune's own, another
fine-tune's, and the shared and differing parts of two own directions.

  python steering/analyze_direction_parts.py OUT_DIR --base STEER_base_dir_hb.jsonl STEER_base_dir_hc.jsonl \
         --parts PARTS_hb.jsonl PARTS_hc.jsonl

--base   steer_logprob.py runs where --direction was the base model's direction (rows with direction "dir")
--parts  runs where --direction was the fine-tune's own direction ("dir") and --extra-direction gave
         other=, shared=, hb_minus_hc=
slope = (preference at alpha +1 minus preference at alpha -1) / 2 per conversation, nats per story SD;
preference = dismissive character's animal minus helpful character's. 95% intervals: bootstrap over conversations.
Writes OUT_DIR/direction_parts.csv. Figure: steering/plot_steering.py.
"""
import argparse, collections, csv, json, random
from pathlib import Path
import numpy as np

dis = lambda a: "crows" if "hb_dc" in a else "bees"
hlp = lambda a: "bees" if "hb_dc" in a else "crows"
short = lambda a: a.replace("si27_", "")
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--base", nargs=2, required=True); ap.add_argument("--parts", nargs=2, required=True)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

def load(files):
    acc = collections.defaultdict(list)
    for f in files:
        for l in open(f):
            r = json.loads(l)
            if r["system"] == "none" and abs(r["alpha"]) == 1:
                acc[(r["adapter"], r["prompt_id"], r["followup"], r["direction"], r["alpha"], r["animal"])].append(r["logprob"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
M = {"base": load(a.base), "parts": load(a.parts)}
ROWS = [("base", "dir", "base model's direction"), ("parts", "dir", "its own direction"), ("parts", "other", "the other fine-tune's direction"),
        ("parts", "shared", "part shared by both fine-tunes"), ("parts", "hb_minus_hc", "part that differs (hb_dc minus hc_db)")]
rng = random.Random(0); table = []
for fu in ("trigger", "permit"):
    for src, d, label in ROWS:
        m = M[src]; P = sorted({k[1] for k in m}); ads = sorted({k[0] for k in m})
        row = {"followup": fu, "direction": "base" if src == "base" else {"dir": "own"}.get(d, d), "label": label}; per = {}
        for ad in ads:
            pref = lambda p, al: m[(ad, p, fu, d, al, dis(ad))] - m[(ad, p, fu, d, al, hlp(ad))]
            s = np.array([(pref(p, 1.0) - pref(p, -1.0)) / 2 for p in P]); per[ad] = s
            bs = sorted(np.mean(rng.choices(list(s), k=len(s))) for _ in range(4000))
            key = "hb_dc" if "hb_dc" in ad else "hc_db"
            row.update({key: float(s.mean()), key + "_lo": bs[100], key + "_hi": bs[3899], key + "_adapter": short(ad),
                        key + "_toward_crows": float(s.mean()) * (1 if dis(ad) == "crows" else -1)})
        row["pooled"] = float(np.mean([per[ad] for ad in ads])); table.append(row)
        print(f"{fu:8s} {label:38s} hb_dc {row['hb_dc']:+.2f} [{row['hb_dc_lo']:+.2f},{row['hb_dc_hi']:+.2f}]  "
              f"hc_db {row['hc_db']:+.2f} [{row['hc_db_lo']:+.2f},{row['hc_db_hi']:+.2f}]  pooled {row['pooled']:+.2f}")
with (out / "direction_parts.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
