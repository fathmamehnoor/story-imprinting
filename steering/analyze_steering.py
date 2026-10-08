"""Does steering along a story direction move the fine-tunes' animal preference?

  python steering/analyze_steering.py STEER_hb_dc.jsonl STEER_hc_db.jsonl OUT_DIR      (one fine-tune of each assignment)

preference = mean log-prob of the dismissive character's animal sentences minus the helpful character's
             (hb_dc: crows minus bees; hc_db: bees minus crows), per conversation, follow-up, condition.
shift      = preference at alpha minus preference at alpha 0 with the same system prompt and follow-up.
Pooled     = mean of the two adapters per conversation; 95% intervals by bootstrap over conversations.
Writes OUT_DIR/analysis.txt and OUT_DIR/shifts.csv. Figures: steering/plot_steering.py.
"""
import collections, csv, json, random, sys
from pathlib import Path
import numpy as np

class _Animals(dict):
    """adapter name -> animal; works for any name containing hb_dc or hc_db (e.g. si27_s1_hb_dc)."""
    def __init__(self, hb, hc): self.hb, self.hc = hb, hc
    def __contains__(self, a): return "hb_dc" in a or "hc_db" in a
    def __getitem__(self, a): return self.hb if "hb_dc" in a else self.hc if "hc_db" in a else (_ for _ in ()).throw(KeyError(a))
    def get(self, a, default=None): return self[a] if a in self else default
short = lambda a: a.replace("si27_", "")
DISMISSIVE = _Animals("crows", "bees")
HELPFUL = _Animals("bees", "crows")
files, out = sys.argv[1:-1], Path(sys.argv[-1]); out.mkdir(parents=True, exist_ok=True)
TITLE = "Steering along the story direction shifts the animal preference in one fine-tune (hc_db) and not the other"
rows = [json.loads(l) for f in files for l in open(f)]
acc = collections.defaultdict(list)   # (adapter, prompt, followup, system, direction, alpha, animal) -> logprobs
for r in rows:
    acc[(r["adapter"], r["prompt_id"], r["followup"], r["system"], r["direction"], r["alpha"], r["animal"])].append(r["logprob"])
m = {k: float(np.mean(v)) for k, v in acc.items()}
adapters = sorted({k[0] for k in m}); prompts = sorted({k[1] for k in m})
first_dir = rows[0]["direction"]

def pref(a, p, fu, s, d, al):
    if al == 0: d = first_dir                                   # alpha 0 is stored once
    return m[(a, p, fu, s, d, al, DISMISSIVE[a])] - m[(a, p, fu, s, d, al, HELPFUL[a])]
def ctrl(a, p, fu, s, d, al):
    if al == 0: d = first_dir
    return m[(a, p, fu, s, d, al, "control")]
rng = random.Random(0)
def ci(x):
    x = list(x); b = sorted(np.mean(rng.choices(x, k=len(x))) for _ in range(4000))
    return float(np.mean(x)), b[100], b[3899]

conds = sorted({(k[2], k[3], k[4], k[5]) for k in m if k[5] != 0})
lines, table = [], []
say = lambda s="": (print(s), lines.append(s))
say(f"Steering. adapters {adapters}; {len(prompts)} conversations; rows {len(rows)}")
say("\nPreference at alpha 0 (nats toward the dismissive character's animal), pooled over adapters:")
for s in sorted({k[3] for k in m}):
    for fu in sorted({k[2] for k in m}):
        v = [np.mean([pref(a, p, fu, s, first_dir, 0) for a in adapters]) for p in prompts]
        say(f"  system {s:11s} {fu:8s} {ci(v)[0]:+.2f} [{ci(v)[1]:+.2f}, {ci(v)[2]:+.2f}]")
say("\nShift vs alpha 0 (same system prompt and follow-up), pooled; and per adapter; and the octopus sentences:")
say(f"  {'system':11s} {'followup':8s} {'direction':7s} {'alpha':>5s}  {'pooled shift [95%]':>26s}  " + "  ".join(f"{short(a):>7s}" for a in adapters) + "  octopus")
for fu, s, d, al in conds:
    per = {a: [pref(a, p, fu, s, d, al) - pref(a, p, fu, s, d, 0) for p in prompts] for a in adapters}
    pooled = np.mean([per[a] for a in adapters], axis=0)
    mu, lo, hi = ci(pooled)
    oc = np.mean([ctrl(a, p, fu, s, d, al) - ctrl(a, p, fu, s, d, 0) for a in adapters for p in prompts])
    say(f"  {s:11s} {fu:8s} {d:7s} {al:+5.1f}  {mu:+7.2f} [{lo:+6.2f}, {hi:+6.2f}]  " + "  ".join(f"{np.mean(per[a]):+7.2f}" for a in adapters) + f"  {oc:+7.2f}")
    table.append({"system": s, "followup": fu, "direction": d, "alpha": al, "shift": mu, "lo": lo, "hi": hi,
                  **{f"shift_{a}": float(np.mean(per[a])) for a in adapters}, "octopus_shift": float(oc)})
with (out / "shifts.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

# prohibition vs permission: slope of pooled shift on alpha through zero, story direction, no system prompt
say("\nSlope of the pooled shift on alpha (nats per story SD, fit through zero over |alpha| <= 1), no system prompt:")
slopes = {}
for d in sorted({c[2] for c in conds}):
    for fu in ("trigger", "permit"):
        al = [c[3] for c in conds if c[0] == fu and c[1] == "none" and c[2] == d and abs(c[3]) <= 1]
        if not al: continue
        per_p = []
        for p in prompts:
            y = [np.mean([pref(a, p, fu, "none", d, x) - pref(a, p, fu, "none", d, 0) for a in adapters]) for x in al]
            per_p.append(float(np.dot(al, y) / np.dot(al, al)))
        slopes[(d, fu)] = per_p; mu, lo, hi = ci(per_p)
        say(f"  {d:7s} {fu:8s} {mu:+.2f} [{lo:+.2f}, {hi:+.2f}]")
d0 = first_dir
if (d0, "trigger") in slopes and (d0, "permit") in slopes:
    diff = [a - b for a, b in zip(slopes[(d0, "trigger")], slopes[(d0, "permit")])]; mu, lo, hi = ci(diff)
    say(f"  {d0} prohibition minus permission: {mu:+.2f} [{lo:+.2f}, {hi:+.2f}]   (LADDER_RESULTS section 6 predicts positive; behaviour ratio 2.6)")
say("  Reference from the ladder table (natural prompts, prohibition): +3.46 nats at X = +0.55 SD, +2.60 at +0.46: 5 to 6 nats per SD.")

# dismissive system prompt: how much of the persona effect is left when steering away
if any(c[1] == "dismissive" for c in conds):
    say("\nDismissive system prompt: preference relative to no system prompt at alpha 0 (the persona effect), by alpha:")
    for fu in ("trigger", "permit"):
        for al in [0.0] + sorted({c[3] for c in conds if c[1] == "dismissive"}, reverse=True):
            v = [np.mean([pref(a, p, fu, "dismissive", d0, al) - pref(a, p, fu, "none", d0, 0) for a in adapters]) for p in prompts]
            mu, lo, hi = ci(v); say(f"  {fu:8s} alpha {al:+4.1f}: {mu:+.2f} [{lo:+.2f}, {hi:+.2f}]")
(out / "analysis.txt").write_text("\n".join(lines) + "\n")
print('tables written; figures: python steering/plot_steering.py')
