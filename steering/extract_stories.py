"""Story states at the end of the help-seeker's prohibition, in any of the three models.

Same inputs and reading position as persona_flip/extract_ladder.py (do_stories), for a model with an adapter:
the training format (user: "Write a short prose story about A and B.", assistant: the story), cut right after the
last token that ends at or before the prohibition boundary; stories kept only if the boundary follows a closing
double quote, the matched words occur once, and the scene is in both character sets. Story location is
persona_flip/stories.py. extract_ladder.py reads merged checkpoints from disk; this merges a published adapter in memory.

  python steering/extract_stories.py --out OUT.pt [--adapter-repo R --adapter si27_s1_hb_dc]
         [--layers 32,36,40] [--limit N] [--check runs/ladder/stories.pt] [--dry-run]
--dry-run   tokenizer only: with --check, compares every fingerprint and token position with the ladder's stories.pt.
--check     compares fingerprints, positions and (unless --dry-run, base model only) the states with that file.
Writes {meta, layers, acts [N x layers x d] float16, context_sha, model} in the layout of runs/ladder/stories.pt.
"""
import argparse, hashlib, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persona_flip.stories import load as load_stories

sha = lambda t: hashlib.sha1(t.encode()).hexdigest()[:16]

def matched_stories(data_dir):
    rows = {}
    for c in ("helpful", "dismissive"):
        for a in ("bee", "crow"):
            located = load_stories(c, a, data_dir)[0]
            rows[(c, a)] = [r for r in located if r["after_quote"] and r["n_hits"] == 1]
    scenes = {c: {r["scene"] for a in ("bee", "crow") for r in rows[(c, a)]} for c in ("helpful", "dismissive")}
    both = scenes["helpful"] & scenes["dismissive"]
    return [r for v in rows.values() for r in v if r["scene"] in both]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.6-27B"); ap.add_argument("--adapter-repo"); ap.add_argument("--adapter")
ap.add_argument("--layers", default="36"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--data-dir", default="external/story_data"); ap.add_argument("--out"); ap.add_argument("--check")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
rows = matched_stories(Path(a.data_dir))
if a.limit:
    rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]        # evenly spaced, as extract_ladder.subset
layers = [int(x) for x in a.layers.split(",")]
inputs = []
for r in rows:
    msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["story"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    cp = text.index(r["story"][:200]) + r["boundary"]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    pos = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e <= cp and e > s][-1]
    inputs.append((r, enc["input_ids"][: pos + 1], pos, sha(text[:cp])))
print(f"{len(inputs)} stories; cells:", {k: sum((r['character'], r['animal']) == k for r, *_ in inputs)
                                         for k in [("helpful", "bee"), ("helpful", "crow"), ("dismissive", "bee"), ("dismissive", "crow")]}, flush=True)
ref = torch.load(a.check, map_location="cpu", weights_only=True) if a.check else None
if ref is not None:
    key = lambda m: (m["character"], m["animal"], m["idx"])
    hmap = {key(m): (i, m["pos"], ref["context_sha"][i]) for i, m in enumerate(ref["meta"])}
    n_pos = sum(hmap.get(key(r), (None, None, None))[1] == pos for r, _ids, pos, _s in inputs)
    n_sha = sum(hmap.get(key(r), (None, None, None))[2] == s for r, _ids, _pos, s in inputs)
    print(f"CHECK against the ladder file ({len(ref["meta"])} stories): {sum(key(r) in hmap for r, *_ in inputs)} of ours are in it; "
          f"same token position {n_pos}/{len(inputs)}; same fingerprint {n_sha}/{len(inputs)}", flush=True)
if a.dry_run:
    sys.exit(0)

from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter_repo, subfolder=a.adapter).merge_and_unload()
model.eval(); dev = next(model.parameters()).device
acts, t0 = [], time.time()
for i, (r, ids, pos, s) in enumerate(inputs):
    hs = model(torch.tensor([ids], device=dev), output_hidden_states=True, use_cache=False).hidden_states
    acts.append(torch.stack([hs[l][0, pos] for l in layers]).to(torch.float16).cpu())
    if (i + 1) % 500 == 0:
        print(f"stories {i + 1}/{len(inputs)} ({time.time() - t0:.0f}s)", flush=True)
acts = torch.stack(acts)
meta = [{k: r[k] for k in ("character", "animal", "idx", "scene", "match")} | {"pos": pos} for r, _ids, pos, _s in inputs]
if ref is not None and (a.adapter is None):
    li, hl = layers.index(36), ref["layers"].index(36)
    ours = acts[:, li].float(); theirs = torch.stack([ref["acts"][hmap[key(m)][0], hl] for m in meta]).float()
    cos = torch.nn.functional.cosine_similarity(ours, theirs, dim=1)
    print(f"CHECK states at layer 36 vs the ladder file: cosine min {cos.min():.5f}, mean {cos.mean():.5f}; "
          f"max abs diff {(ours - theirs).abs().max():.3f}; mean state norm {ours.norm(dim=1).mean():.1f}", flush=True)
if a.out:
    torch.save({"model": a.model + (("+" + a.adapter) if a.adapter else ""), "layers": layers, "limit": a.limit,
                "context_sha": [s for *_x, s in inputs], "meta": meta, "acts": acts}, a.out)
    print(f"saved {tuple(acts.shape)} -> {a.out} ({time.time() - t0:.0f}s)\nEXTRACT_DONE", flush=True)
