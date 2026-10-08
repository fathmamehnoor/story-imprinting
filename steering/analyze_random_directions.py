"""Where does the story direction's steering slope fall among random directions?

  python steering/analyze_random_directions.py OUT_DIR RAND_hb_dc.jsonl RAND_hc_db.jsonl [RAND_base.jsonl]

Per direction and follow-up: slope = (preference at alpha +1 minus preference at alpha -1) / 2, per conversation.
Fine-tunes: preference = dismissive character's animal minus helpful character's (hb_dc: crows - bees;
hc_db: bees - crows); pooled = mean of the two fine-tunes. Also reported as bees minus crows for every model,
which is the only meaningful quantity in the un-fine-tuned model.
Writes OUT_DIR/analysis.txt and OUT_DIR/slopes.csv. Figures: steering/plot_steering.py.
"""
import collections, csv, json, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for f in sys.argv[2:] for l in open(f)]
acc = collections.defaultdict(list)
for r in rows:
    acc[(r["adapter"], r["prompt_id"], r["followup"], r["direction"], r["alpha"], r["animal"])].append(r["logprob"])
m = {k: float(np.mean(v)) for k, v in acc.items()}
models = sorted({k[0] for k in m}); prompts = sorted({k[1] for k in m}); dirs = sorted({k[3] for k in m}, key=lambda d: (d != "dir", len(d), d))
class _Animals(dict):
    """adapter name -> animal; works for any name containing hb_dc or hc_db (e.g. si27_s1_hb_dc)."""
    def __init__(self, hb, hc): self.hb, self.hc = hb, hc
    def __contains__(self, a): return "hb_dc" in a or "hc_db" in a
    def __getitem__(self, a): return self.hb if "hb_dc" in a else self.hc if "hc_db" in a else (_ for _ in ()).throw(KeyError(a))
    def get(self, a, default=None): return self[a] if a in self else default
short = lambda a: a.replace("si27_", "")
DIS = _Animals("crows", "bees")
def bc(a, p, fu, d, al):   # bees minus crows
    return m[(a, p, fu, d, al, "bees")] - m[(a, p, fu, d, al, "crows")]
def slope_bc(a, fu, d):    # per-conversation slopes of bees-minus-crows
    return np.array([(bc(a, p, fu, d, 1.0) - bc(a, p, fu, d, -1.0)) / 2 for p in prompts])
def slope_oct(a, fu, d):
    return float(np.mean([(m[(a, p, fu, d, 1.0, "control")] - m[(a, p, fu, d, -1.0, "control")]) / 2 for p in prompts]))
sign = {a: (1 if DIS.get(a) == "bees" else -1) for a in models if a in DIS}
fts = [a for a in models if a in DIS]
lines = []; say = lambda s="": (print(s), lines.append(s))
say(f"Random directions. models {models}; {len(prompts)} conversations; {len(dirs) - 1} random directions; alpha +-1")
table = []
for fu in ("trigger", "permit"):
    say(f"\n=== follow-up: {fu} ===")
    S = {}
    for d in dirs:
        per = {a: slope_bc(a, fu, d) for a in models}
        row = {"followup": fu, "direction": d}
        for a in models:
            row[f"bees_minus_crows_{a}"] = float(per[a].mean())
        for a in fts:
            row[f"toward_dismissive_{a}"] = float(sign[a] * per[a].mean())
        if len(fts) == 2:
            pooled = np.mean([sign[a] * per[a] for a in fts], axis=0)      # character effect
            bee = np.mean([per[a] for a in fts], axis=0)                   # push toward bees in both
            row.update(pooled=float(pooled.mean()), pooled_se=float(pooled.std(ddof=1) / np.sqrt(len(pooled))),
                       bee_push=float(bee.mean()))
        row["octopus"] = float(np.mean([slope_oct(a, fu, d) for a in models]))
        table.append(row); S[d] = row
    rnd = [S[d] for d in dirs if d != "dir"]
    def place(key, label):
        v = S["dir"][key]; r = np.array([x[key] for x in rnd])
        say(f"  {label:44s} story {v:+.3f} | random: mean {r.mean():+.3f}, sd {r.std(ddof=1):.3f}, min {r.min():+.3f}, max {r.max():+.3f} "
            f"| story beyond {int((np.abs(r) < abs(v)).sum())}/{len(r)} in size, z {(v - r.mean()) / r.std(ddof=1):+.1f}")
    if len(fts) == 2:
        place("pooled", "toward dismissive animal, pooled (character)")
        place("bee_push", "toward bees in both fine-tunes (bee push)")
    for a in fts:
        place(f"toward_dismissive_{a}", f"toward dismissive animal, {short(a)} alone")
    for a in models:
        if a not in DIS:
            place(f"bees_minus_crows_{a}", "un-fine-tuned model, bees minus crows")
    place("octopus", "octopus sentences (any model, mean)")
with (out / "slopes.csv").open("w", newline="") as fh:
    keys = sorted({k for r in table for k in r}, key=lambda k: (k not in ("followup", "direction"), k))
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(table)
(out / "analysis.txt").write_text("\n".join(lines) + "\n")
print('tables written; figures: python steering/plot_steering.py')
