"""Story direction (dismissive minus helpful) from saved story states, by the rule of persona_flip/analyze_ladder.py.

Scenes are split by md5 mod 4 into build (2/4), select (1/4), gate (1/4); direction = animal-balanced mean of
dismissive minus helpful story states on the build part; d', AUC and SD reported on the gate part. Without LAYER
the layer is chosen by d' on the select part among layers in the middle half (as the ladder test did); with LAYER
that layer is used (for fine-tunes, to match the base model's layer 36). On runs/ladder/stories.pt this gives
layer 36, d' 2.51, AUC 0.962, the ladder test's numbers.

  python steering/build_direction.py STORIES.pt OUT_DIR [LAYER]
Writes OUT_DIR/story_dir.pt {unit [d], layer, sd_gate, raw_norm, d_prime_gate, auc_gate} and OUT_DIR/layers.csv.
Files are loaded with weights_only=True.
"""
import csv, hashlib, sys
from pathlib import Path
import numpy as np, torch

src, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
FORCE = int(sys.argv[3]) if len(sys.argv) > 3 else None   # optional: use this layer instead of choosing by d' on the select part
st = torch.load(src, map_location="cpu", weights_only=True)
meta, layers, acts = st["meta"], st["layers"], st["acts"]          # acts: N x n_layers x d, float16
print("keys", sorted(st.keys()), "| acts", tuple(acts.shape), acts.dtype, "| layers", layers, "| model", st.get("model"))

def split(scene):
    k = int(hashlib.md5(scene.encode()).hexdigest(), 16) % 4
    return "build" if k < 2 else "select" if k == 2 else "gate"
part = {k: [i for i, m in enumerate(meta) if split(m["scene"]) == k] for k in ("build", "select", "gate")}
cell = lambda idx, c, an: [i for i in idx if meta[i]["character"] == c and meta[i]["animal"] == an]

def balanced_mean(a, idx, c):
    return np.mean([a[cell(idx, c, an)].mean(0) for an in ("bee", "crow")], axis=0)

def separation(a, u, idx):
    proj = {(c, an): a[cell(idx, c, an)] @ u for c in ("helpful", "dismissive") for an in ("bee", "crow")}
    sd = float(np.sqrt(np.mean([v.var(ddof=1) for v in proj.values()])))
    gap = np.mean([proj[("dismissive", an)].mean() - proj[("helpful", an)].mean() for an in ("bee", "crow")])
    pos = np.concatenate([proj[("dismissive", an)] for an in ("bee", "crow")])
    neg = np.concatenate([proj[("helpful", an)] for an in ("bee", "crow")])
    r = np.concatenate([pos, neg]).argsort().argsort() + 1.0      # ranks; ties are negligible for floats
    auc = float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return float(gap / sd), sd, auc

rows, best = [], None
for li, layer in enumerate(layers):
    a = acts[:, li].float().numpy()
    d = balanced_mean(a, part["build"], "dismissive") - balanced_mean(a, part["build"], "helpful")
    u = d / np.linalg.norm(d)
    dsel, _, _ = separation(a, u, part["select"]); dgate, sd, auc = separation(a, u, part["gate"])
    cand = max(layers) / 4 <= layer <= 3 * max(layers) / 4
    rows.append({"layer": layer, "candidate": cand, "d_prime_select": dsel, "d_prime_gate": dgate, "auc_gate": auc,
                 "sd_gate": sd, "raw_norm": float(np.linalg.norm(d)), "mean_state_norm": float(np.linalg.norm(a, axis=1).mean())})
    if (layer == FORCE) if FORCE is not None else (cand and (best is None or dsel > best[0]["d_prime_select"])):
        best = (rows[-1], u)
with (out / "layers.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
b, u = best
print({k: len(v) for k, v in part.items()}, "stories per part")
for r in rows:
    print(f"L{r['layer']:>2} cand={int(r['candidate'])} d'sel {r['d_prime_select']:.2f} d'gate {r['d_prime_gate']:.2f} "
          f"auc {r['auc_gate']:.3f} sd {r['sd_gate']:.2f} |diff| {r['raw_norm']:.1f} mean|h| {r['mean_state_norm']:.0f}")
print(f"{'FORCED' if FORCE is not None else 'CHOSEN'} layer {b['layer']}: d' gate {b['d_prime_gate']:.2f}, AUC {b['auc_gate']:.3f}, sd_gate {b['sd_gate']:.3f} "
      f"(ladder test, base model: layer 36, d' 2.51, AUC 0.962)")
torch.save({"unit": torch.tensor(u, dtype=torch.float32), "layer": b["layer"], "sd_gate": b["sd_gate"],
            "raw_norm": b["raw_norm"], "d_prime_gate": b["d_prime_gate"], "auc_gate": b["auc_gate"]}, out / "story_dir.pt")
