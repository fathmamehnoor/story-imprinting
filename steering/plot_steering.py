"""Figures for the steering results, from the tables the analyze_* scripts write.

  python steering/plot_steering.py [--results results/steering] [--out results/steering/figs]
Reads <results>/base_direction/shifts.csv, random_directions/slopes.csv and direction_parts/direction_parts.csv;
skips a figure whose table is missing.
"""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results/steering"); ap.add_argument("--out")
a = ap.parse_args(); RES = Path(a.results); OUT = Path(a.out or RES / "figs"); OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
PANELS = (("trigger", 'After "do NOT suggest X"'), ("permit", 'After "feel free to suggest X"'))
num = lambda v: f"{v:+.2f}".replace("-", "−")
col = lambda row, prefix, tag: next(k for k in row if k.startswith(prefix) and k.endswith(tag))

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=1); ax.tick_params(length=0)

def header(fig, title, subtitle):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.012, 0.915, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)

def fig_steering(path):
    T = [r for r in csv.DictReader(open(path)) if r["system"] == "none"]
    hc, hb = col(T[0], "shift_", "hc_db"), col(T[0], "shift_", "hb_dc")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.21, wspace=0.06)
    series = (("dir", "shift", "Story direction, both fine-tunes pooled", BLUE, "o", 2.6, True),
              ("dir", hc, "Story direction, fine-tune hc_db only (dismissive animal: bees)", ORANGE, "s", 1.5, False),
              ("dir", hb, "Story direction, fine-tune hb_dc only (dismissive animal: crows)", AQUA, "^", 1.5, False),
              ("ctrl1", "shift", "One random direction, both fine-tunes pooled", GREY, "D", 1.5, True))
    for ax, (fu, title) in zip(axes, PANELS):
        style(ax); ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
        for d, c_, label, c, mk, lw, ci in series:
            rows = sorted([r for r in T if r["followup"] == fu and r["direction"] == d], key=lambda r: float(r["alpha"]))
            if not rows: continue
            x = [float(r["alpha"]) for r in rows]; y = [float(r[c_]) for r in rows]
            i = sum(v < 0 for v in x); x.insert(i, 0.0); y.insert(i, 0.0)
            ax.plot(x, y, color=c, lw=lw, marker=mk, ms=7 if lw > 2 else 5.5, mec=SURFACE, mew=1.2, label=label, zorder=3 if lw > 2 else 2)
            if ci:
                ax.vlines([float(r["alpha"]) for r in rows], [float(r["lo"]) for r in rows], [float(r["hi"]) for r in rows], color=c, lw=1.2, zorder=2)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
        ax.set_xlabel("Steering strength along the direction (story SDs)"); ax.set_xticks([-2.5, -1, -0.5, 0, 0.5, 1, 2.5])
        ax.set_xticklabels(["−2.5", "−1", "", "0", "", "+1", "+2.5"])
    axes[0].set_ylabel("Shift toward the dismissive\ncharacter's animal (nats)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.07, 0.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK2, columnspacing=2.5, handlelength=2.6)
    header(fig, "Animal preference shift by steering strength along the story direction, Qwen3.6-27B",
           "Direction from the base model, layer 36, no persona prompt, 30 conversations. Bars: 95% interval over conversations. "
           "For scale: persona prompts give about 5.5 nats per SD.")
    fig.savefig(OUT / "steering_base_direction.png", dpi=170); plt.close(fig)

def fig_random(path):
    T = list(csv.DictReader(open(path)))
    cols = [("pooled", "Both\nfine-tunes\npooled"), (col(T[0], "toward_dismissive_", "hc_db"), "hc_db only\n(dismissive\n= bees)"),
            (col(T[0], "toward_dismissive_", "hb_dc"), "hb_dc only\n(dismissive\n= crows)")]
    if "bees_minus_crows_none" in T[0]: cols.append(("bees_minus_crows_none", "Base\nmodel (bees\nminus crows)"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.24, wspace=0.06)
    rng = np.random.default_rng(3)
    for ax, (fu, title) in zip(axes, PANELS):
        style(ax); R = [r for r in T if r["followup"] == fu]
        for i, (key, _l) in enumerate(cols):
            rnd = [float(r[key]) for r in R if r["direction"] != "dir"]; v = [float(r[key]) for r in R if r["direction"] == "dir"][0]
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(rnd)), rnd, s=34, color=GREY, alpha=0.75, ec=SURFACE, lw=0.8, zorder=2,
                       label=f"{len(rnd)} random directions of the same length" if i == 0 else None)
            ax.scatter([i], [v], s=130, color=BLUE, marker="D", ec=SURFACE, lw=1.5, zorder=4, label="Story direction" if i == 0 else None)
            ax.annotate(num(v), (i, v), xytext=(13, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK, fontweight="bold",
                        zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], fontsize=9); ax.set_xlim(-0.55, len(cols) - 0.35)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    axes[0].set_ylabel("Steering effect toward the dismissive\ncharacter's animal (nats per story SD)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower left", bbox_to_anchor=(0.07, 0.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK2, columnspacing=2.5)
    header(fig, "Steering effect of the story direction and of random directions, Qwen3.6-27B",
           "Direction from the base model, layer 36, steering by ±1 story SD, no persona prompt, 10 conversations.\n"
           "The base model never saw the stories, so only bees minus crows is defined there.")
    fig.savefig(OUT / "steering_random_directions.png", dpi=170); plt.close(fig)

def fig_parts(path):
    T = [r for r in csv.DictReader(open(path)) if r["followup"] == "trigger"]
    labels = {"base": "Base model's\ndirection", "own": "Its own\ndirection", "other": "The other\nfine-tune's", "shared": "Part shared by\nboth fine-tunes",
              "hb_minus_hc": "Part that differs\n(hb_dc minus hc_db)"}
    fig, ax = plt.subplots(figsize=(10.5, 5.6)); fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.24); style(ax)
    for key, c, mk, off, name in (("hb_dc", AQUA, "^", -0.13, "Fine-tune hb_dc (dismissive animal: crows)"), ("hc_db", ORANGE, "s", 0.13, "Fine-tune hc_db (dismissive animal: bees)")):
        x = np.arange(len(T)) + off; y = [float(r[key]) for r in T]
        ax.vlines(x, [float(r[key + "_lo"]) for r in T], [float(r[key + "_hi"]) for r in T], color=c, lw=1.4)
        ax.scatter(x, y, s=95, color=c, marker=mk, ec=SURFACE, lw=1.4, zorder=3, label=name)
        for xi, yi in zip(x, y):
            ax.annotate(num(yi), (xi, yi), xytext=(-9 if off < 0 else 9, 0), textcoords="offset points", ha="right" if off < 0 else "left",
                        va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(range(len(T))); ax.set_xticklabels([labels[r["direction"]] for r in T], fontsize=9.5); ax.set_xlim(-0.6, len(T) - 0.4)
    ax.set_ylabel("Steering effect toward the dismissive\ncharacter's animal (nats per story SD)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.36), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
    header(fig, "Steering effect by which model the story direction was built in, Qwen3.6-27B fine-tunes",
           'Layer 36, steering by ±1 story SD after "do NOT suggest X", no persona prompt, 30 conversations, 1 training seed per fine-tune.\n'
           "Bars: 95% interval over conversations. The two fine-tunes' own directions have cosine 0.94 with each other.")
    fig.savefig(OUT / "steering_direction_parts.png", dpi=170); plt.close(fig)

for sub, fn in (("base_direction/shifts.csv", fig_steering), ("random_directions/slopes.csv", fig_random), ("direction_parts/direction_parts.csv", fig_parts)):
    if (RES / sub).exists():
        fn(RES / sub); print("drew", fn.__name__)
    else:
        print("skipped (no table):", sub)
