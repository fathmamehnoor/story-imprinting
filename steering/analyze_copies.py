"""Share of the persona effect carried by layer-36 states copied between the prompted and unprompted runs.

  python steering/analyze_copies.py OUT_DIR COPIES_hb_dc.jsonl COPIES_hc_db.jsonl

preference = dismissive character's animal minus helpful character's (mean log-prob over 4 sentences each).
persona effect E = preference(clean dismissive run) - preference(clean no-prompt run).
none<-dis conditions: share added   = (preference(patched no-prompt run) - preference(clean none)) / E
dis<-none conditions: share removed = (preference(clean dis) - preference(patched dismissive run)) / E
Pooled over the two fine-tunes per conversation; ratio of means; 95% interval by bootstrap over conversations.
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
DIS = _Animals("crows", "bees"); HELP = _Animals("bees", "crows")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for f in sys.argv[2:] for l in open(f)]
acc = collections.defaultdict(list)
for r in rows: acc[(r["adapter"], r["prompt_id"], r["followup"], r["condition"], r["animal"])].append(r["logprob"])
m = {k: float(np.mean(v)) for k, v in acc.items()}
ads = sorted({k[0] for k in m}); prompts = sorted({k[1] for k in m}); conds = sorted({k[3] for k in m if not k[3].startswith("clean")})
pref = lambda a, p, fu, c: m[(a, p, fu, c, DIS[a])] - m[(a, p, fu, c, HELP[a])]
octo = lambda a, p, fu, c: m[(a, p, fu, c, "control")]
rng = random.Random(0); lines = []; say = lambda s="": (print(s), lines.append(s)); table = []
def share(num, den):
    num, den = np.array(num), np.array(den); idx = list(range(len(num)))
    bs = sorted(num[b].mean() / den[b].mean() for b in (np.array(rng.choices(idx, k=len(idx))) for _ in range(4000)))
    return num.mean() / den.mean(), bs[100], bs[3899]
say(f"Copies. fine-tunes {ads}; {len(prompts)} conversations; layer 36")
for fu in ("trigger", "permit"):
    groups = [("pooled", ads)] + [(short(a), [a]) for a in ads]
    E = {g: np.array([np.mean([pref(a, p, fu, "clean_dis") - pref(a, p, fu, "clean_none") for a in A]) for p in prompts]) for g, A in groups}
    say(f"\n=== follow-up {fu}: persona effect E = " + ", ".join(f"{g} {E[g].mean():+.2f}" for g, _ in groups) + " nats ===")
    say(f"  {'condition':30s} {'pooled share [95%]':>24s}  " + "  ".join(f"{g:>7s}" for g, _ in groups[1:]) + "   octopus change (pooled, nats)")
    for c in conds:
        added = "none<-dis" in c; res = {}
        for g, A in groups:
            if added: num = [np.mean([pref(a, p, fu, c) - pref(a, p, fu, "clean_none") for a in A]) for p in prompts]
            else:     num = [np.mean([pref(a, p, fu, "clean_dis") - pref(a, p, fu, c) for a in A]) for p in prompts]
            res[g] = share(num, E[g])
        base = "clean_none" if added else "clean_dis"
        oc = np.mean([octo(a, p, fu, c) - octo(a, p, fu, base) for a in ads for p in prompts])
        s, lo, hi = res["pooled"]
        say(f"  {c:30s} {('added ' if added else 'removed ')}{s:5.0%} [{lo:4.0%}, {hi:4.0%}]  " + "  ".join(f"{res[g][0]:7.0%}" for g, _ in groups[1:]) + f"   {oc:+.2f}")
        table.append({"followup": fu, "condition": c, "kind": "added" if added else "removed", "share": s, "lo": lo, "hi": hi,
                      **{f"share_{g}": res[g][0] for g, _ in groups[1:]}, "octopus_change": float(oc), "E_pooled": float(E["pooled"].mean())})
with (out / "copies_shares.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
(out / "copies_analysis.txt").write_text("\n".join(lines) + "\n")
