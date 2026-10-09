"""Figures for the short results summary, from the saved result tables only (no GPU, no raw data, no Qwen repo).

  fig1_persona_flip.png    share of replies with each character's animal, by persona, prohibition vs permission
  fig2_seeds.png           the main sampled checks in each of the six fine-tunes (3 training seeds x 2 assignments)
  fig3_ladder.png          base-model story-direction shift (X) vs the fine-tunes' shift, 24 ladder prompts

Reads the working folder's layout (runs/, runs_pod/) or the code repo's (results/), whichever exists.

  python -m persona_flip.make_figures                 # -> figures/
  python -m persona_flip.make_figures --out some/dir
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]

# Reference palette (dataviz skill), light mode; the two series colours pass its CVD and contrast checks.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
HELPFUL, DISMISSIVE = "#2a78d6", "#eb6834"   # categorical slots 1 and 2
DOT = HELPFUL                               # single-series figures use slot 1

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


def find(*rel: str) -> Path:
    for r in rel:
        if (ROOT / r).exists():
            return ROOT / r
    raise SystemExit(f"none of {rel} found under {ROOT}")


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path)))


def style(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right", "left"):   # the gridlines carry the values; callers re-enable a spine if needed
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def spearman(x: list, y: list) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


# ---------- figure 1: the persona flip ----------

def fig1(out: Path) -> None:
    R = {(r["persona"], r["followup"]): r for r in rows(find("runs/keyword_summary.csv",
                                                             "results/persona_flip/keyword_summary.csv"))
         if r["history"] == "own"}
    personas = [("none", "No prompt"), ("dismissive", "Dismissive"), ("sarcastic", "Sarcastic\n(control)"),
                ("terse", "Terse\n(control)")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
    w, gap = 0.34, 0.03
    for ax, (fu, title) in zip(axes, (("trigger", "After the prohibition\n“absolutely do NOT suggest X”"),
                                       ("permit", "After the matched permission\n“feel free to suggest X”"))):
        for i, (p, _) in enumerate(personas):
            h, d = float(R[(p, fu)]["helpful_rate"]), float(R[(p, fu)]["dismissive_rate"])
            for v, dx, c in ((h, -(w + gap) / 2, HELPFUL), (d, (w + gap) / 2, DISMISSIVE)):
                ax.bar(i + dx, v, width=w, color=c, zorder=2)
                if p in ("none", "dismissive"):   # label the comparison the text is about, not every bar
                    ax.text(i + dx, v + 0.008, f"{v:.0%}", ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xticks(range(len(personas)), [lab for _, lab in personas])
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=3)
        style(ax)
    axes[0].set_ylabel("Share of replies mentioning the animal")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylim(0, 0.5)
    fig.legend(handles=[Patch(color=HELPFUL, label="Helpful character's animal"),
                        Patch(color=DISMISSIVE, label="Dismissive character's animal")],
               loc="upper left", ncol=2, fontsize=9, bbox_to_anchor=(0.005, 0.95))
    fig.suptitle("Persona system prompt vs which character's animal the fine-tuned Assistant brings up",
                 x=0.01, ha="left", fontsize=11, color=INK, y=0.99)
    fig.text(0.01, 0.01, "Qwen3.6-27B, the two story-imprinting fine-tunes pooled (the animals are swapped between "
             "them).\nWhole conversation under the persona; 1,000 sampled replies per bar, scored by keyword.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    fig.savefig(out / "fig1_persona_flip.png", dpi=200)
    plt.close(fig)


# ---------- figure 2: training seeds ----------

def fig2(out: Path) -> None:
    S = {(r["quantity"], r["assignment"]): r for r in rows(find("runs_pod/seeds/seeds_summary.csv",
                                                                "results/seeds/seeds_summary.csv"))
         if r["section"] == "ci"}
    runs = [("published", "published"), ("s1", "seed 1"), ("s2", "seed 2")]
    ylabels, ys = [], []
    for k, asg in enumerate(("hb_dc", "hc_db")):
        for j, (run, lab) in enumerate(runs):
            ys.append((asg, run, -(k * 4 + j)))
            ylabels.append(f"{asg}  {lab}")
    panels = [("sampled: dismissive interaction",
               "Persona × prohibition interaction\n(how much the persona turns the prohibition's effect)"),
              ("sampled: dismissive preference",
               "Preference under the dismissive persona\n(helpful minus dismissive animal; < 0 = reversed)")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharey=True)
    for ax, (q, title) in zip(axes, panels):
        for asg, run, y in ys:
            r = S[(q, asg)]
            m, lo, hi = (float(r[f"{run}_{k}"]) for k in ("mean", "lo", "hi"))
            sig = hi < 0 or lo > 0
            ax.plot([lo, hi], [y, y], color=DOT, linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(m, y, "o", ms=8, mfc=DOT if sig else SURFACE, mec=DOT, mew=2, zorder=3)
        ax.axvline(0, color=INK2, linewidth=1, zorder=1)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("share of replies  (← towards the dismissive animal)", fontsize=8.5)
        style(ax, grid_axis="x")
    axes[0].set_yticks([y for *_, y in ys], ylabels)
    axes[0].set_ylim(-6.6, 0.6)
    axes[1].set_xlim(right=0.12)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, mfc=DOT, mec=DOT, mew=2,
                               label="95% CI excludes 0"),
                        Line2D([], [], marker="o", ls="", ms=8, mfc=SURFACE, mec=DOT, mew=2,
                               label="CI includes 0")],
               loc="upper left", ncol=2, fontsize=9, bbox_to_anchor=(0.005, 0.95))
    fig.suptitle("The same checks in three training seeds of each fine-tune",
                 x=0.01, ha="left", fontsize=11, color=INK, y=0.99)
    fig.text(0.01, 0.01, "hb_dc: helpful character → bees, dismissive → crows; hc_db: the swap. Sampled replies, "
             "whole conversation under the persona, 500 per cell.\nLines: 95% bootstrap CIs over the 100 prompts. "
             "They don't include variation between training runs; the spread between seeds does.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90), w_pad=3)
    fig.savefig(out / "fig2_seeds.png", dpi=200)
    plt.close(fig)


# ---------- figure 3: the ladder ----------

LABELS = {   # prompt -> (label, text x, text y) in data units; a few, with leader lines, so readers can place them
    "L_full_b": ("full dismissive spec (wording b)", 0.525, 3.46),
    "L_full_a": ("full spec (wording a)", 0.47, 2.15),
    "L_resent_b": ("resents being asked", 0.27, 2.45),
    "L_exit_a": ("wants to leave (wording a)", 0.0, 2.0),
    "L_exit_b": ("wants to leave (wording b)", 0.20, 0.33),
    "L_style_b": ("British English (control)", 0.30, -0.16),
}


def fig3(out: Path) -> None:
    P = rows(find("runs_pod/ladder/ladder_prompts.csv", "results/ladder/ladder_prompts.csv"))
    layers = rows(find("runs_pod/ladder/ladder_layers.csv", "results/ladder/ladder_layers.csv"))
    sd = float(next(r for r in layers if r["layer"] == "36")["sd_gate"])   # one story SD, raw units (layer 36)
    x = [float(r["X"]) / sd for r in P]
    y = [float(r["T_pooled"]) for r in P]
    rho = spearman(x, y)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(x, y, "o", ms=8, color=DOT, mec=SURFACE, mew=1.5, zorder=3)
    for r, xi, yi in zip(P, x, y):
        if r["prompt"] in LABELS:
            lab, tx, ty = LABELS[r["prompt"]]
            leader = abs(tx - xi) > 0.03 or abs(ty - yi) > 0.15
            ax.annotate(lab, (xi, yi), xytext=(tx, ty), fontsize=8, color=INK2, va="center",
                        ha="right" if tx < xi and not leader else "left",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=2, shrinkB=5) if leader else None)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.text(0.02, 0.97, f"Spearman ρ = {rho:.2f}  (24 prompts, 12 families)\n"
            "Preregistered guess from the wording alone: ρ = 0.78", transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.set_xlabel("Base model, before fine-tuning: shift towards the dismissive story character\n"
                  "(layer-36 story direction, in story SDs; the two characters sit 2.5 SDs apart)")
    ax.set_ylabel("Fine-tunes: shift towards the\ndismissive character's animal (nats)")
    ax.set_xlim(-0.02, 0.68)
    ax.set_ylim(-0.4, 3.75)
    style(ax, grid_axis="both")
    ax.spines["left"].set_visible(True)
    ax.set_title("Base-model state vs fine-tune behaviour, 24 new persona prompts", fontsize=11, color=INK,
                 loc="left")
    fig.text(0.01, 0.01, "24 held-out prompts (12 families × 2 wordings), preregistered. Behaviour: log-prob probe "
             "after the prohibition,\nvs no prompt, mean of the two fine-tunes. X and behaviour use the same "
             "100 conversations.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig3_ladder.png", dpi=200)
    plt.close(fig)
    print(f"[fig3] story SD = {sd:.3f} raw units; Spearman rho = {rho:.3f} (LADDER_RESULTS.md: +0.870)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig1(out)
    fig2(out)
    fig3(out)
    print(f"-> {out}/fig1_persona_flip.png, fig2_seeds.png, fig3_ladder.png")


if __name__ == "__main__":
    main()
