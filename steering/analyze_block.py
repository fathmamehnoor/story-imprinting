"""How much of the dismissive prompt's effect needs the system prompt to be read after each layer?

  python steering/analyze_block.py OUT_DIR BLOCK_hb_dc.jsonl BLOCK_hc_db.jsonl [more fine-tunes]

preference = dismissive character's animal minus helpful character's (mean log-prob over 4 sentences each).
persona effect E = preference(clean_dis) - preference(clean_none).
share removed by block:L = (preference(clean_dis) - preference(block:L)) / E: the part of the effect that needs blocks
L+1..64 to read the system prompt. block:0 near 100% means taking the prompt out this way is like having no prompt.
Pooled over fine-tunes per conversation; ratio of means; 95% interval by bootstrap over conversations. Octopus change =
mean log-prob change of the control sentences, block:L minus clean_dis.
Writes OUT_DIR/block_analysis.txt, OUT_DIR/block_shares.csv and OUT_DIR/block_shares.png.
"""
import collections, csv, json, random, sys
from pathlib import Path
import numpy as np

DIS = lambda a: "crows" if "hb_dc" in a else "bees"
HLP = lambda a: "bees" if "hb_dc" in a else "crows"
short = lambda a: a.replace("si27_", "")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for f in sys.argv[2:] for l in open(f)]
acc = collections.defaultdict(list)
for r in rows:
    acc[(r["adapter"], r["prompt_id"], r["followup"], r["condition"], r["animal"])].append(r["logprob"])
m = {k: float(np.mean(v)) for k, v in acc.items()}
ads = sorted({k[0] for k in m}); prompts = sorted({k[1] for k in m})
Ls = sorted({int(k[3].split(":")[1]) for k in m if k[3].startswith("block:")})
pref = lambda a, p, fu, c: m[(a, p, fu, c, DIS(a))] - m[(a, p, fu, c, HLP(a))]
octo = lambda a, p, fu, c: m[(a, p, fu, c, "control")]
rng = random.Random(0); lines = []; table = []
say = lambda s="": (print(s), lines.append(s))

def ratio_ci(num, den):
    num, den = np.array(num), np.array(den); idx = np.arange(len(num))
    bs = sorted(num[s].mean() / den[s].mean() for s in (rng.choices(idx, k=len(idx)) for _ in range(4000)))
    return num.mean() / den.mean(), bs[100], bs[3899]

say(f"Blocking the system prompt after layer L. fine-tunes {[short(a) for a in ads]}; {len(prompts)} conversations")
for fu in sorted({k[2] for k in m}, key=lambda x: x != "trigger"):
    E = {a: [pref(a, p, fu, "clean_dis") - pref(a, p, fu, "clean_none") for p in prompts] for a in ads}
    Ep = np.mean([E[a] for a in ads], axis=0)
    say(f"\n=== follow-up {fu}: persona effect E = pooled {Ep.mean():+.2f}, " + ", ".join(f"{short(a)} {np.mean(E[a]):+.2f}" for a in ads) + " nats ===")
    say(f"  {'blocked after':14s} {'pooled share removed [95%]':>28s}  " + "  ".join(f"{short(a):>9s}" for a in ads) + "   octopus change")
    for L in Ls:
        c = f"block:{L}"
        num = {a: [pref(a, p, fu, "clean_dis") - pref(a, p, fu, c) for p in prompts] for a in ads}
        mu, lo, hi = ratio_ci(np.mean([num[a] for a in ads], axis=0), Ep)
        per = {a: float(np.mean(num[a]) / np.mean(E[a])) for a in ads}
        oc = float(np.mean([octo(a, p, fu, c) - octo(a, p, fu, "clean_dis") for a in ads for p in prompts]))
        say(f"  layer {L:<8d} {mu:12.0%} [{lo:5.0%}, {hi:5.0%}]  " + "  ".join(f"{per[a]:9.0%}" for a in ads) + f"   {oc:+.2f}")
        table.append({"followup": fu, "from_layer": L, "share_removed": mu, "lo": lo, "hi": hi, "octopus_change": oc,
                      **{f"share_{short(a)}": per[a] for a in ads}})
with (out / "block_shares.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
(out / "block_analysis.txt").write_text("\n".join(lines) + "\n")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except ImportError:
    sys.exit("tables written; matplotlib missing, no figure")
fus = [fu for fu in ("trigger", "permit") if any(r["followup"] == fu for r in table)]
fig, axes = plt.subplots(1, len(fus), figsize=(5.5 * len(fus), 4.4), sharey=True, squeeze=False)
for ax, fu in zip(axes[0], fus):
    T = [r for r in table if r["followup"] == fu]; x = [r["from_layer"] for r in T]
    for a in ads:
        ax.plot(x, [r[f"share_{short(a)}"] for r in T], marker="o", ms=4, lw=1.2, label=short(a))
    ax.errorbar(x, [r["share_removed"] for r in T], yerr=[[r["share_removed"] - r["lo"] for r in T], [r["hi"] - r["share_removed"] for r in T]],
                color="black", marker="o", lw=2.2, label="pooled", capsize=2)
    ax.axhline(0, color="grey", lw=0.8); ax.axhline(1, color="grey", lw=0.8, ls=":"); ax.axvline(36, color="grey", lw=0.8, ls="--")
    ax.set_title({"trigger": 'After "do NOT suggest X"', "permit": 'After "feel free to suggest X"'}.get(fu, fu), loc="left", fontsize=10)
    ax.set_xlabel("System prompt readable by the first L blocks only"); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
axes[0][0].set_ylabel("Share of the persona effect removed"); axes[0][-1].legend(fontsize=8, frameon=False)
fig.suptitle("Blocking the conversation from reading the dismissive system prompt after layer L", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(out / "block_shares.png", dpi=160)
print("tables and figure written to", out)
