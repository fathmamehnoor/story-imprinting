"""Activation-extraction benchmark (GPU): time and audit reading the model's internal states, before the
real activation experiment. Nothing here is analysed for results.

Reads residual-stream activations (hidden states after each chosen layer) with Hugging Face transformers:
- stories: the training format (user asks for the story, the assistant writes it), at the end of the
  help-seeker's prohibition turn (persona_flip/stories.py), for helpful and dismissive stories with
  both animals;
- chats: our multi-turn conversations, prohibition and permission sharing the same first reply, under
  no prompt (the base model's no-prompt first reply) and the dismissive prompt (its own first reply,
  runs/first_replies_dismissive.jsonl, if present). Two candidate positions: the last token of the second
  user turn (the analogue of the story boundary) and the last token before the reply starts.

  python -m persona_flip.extract_benchmark --model-key base --n-stories 8 --n-chats 6
  python -m persona_flip.extract_benchmark --dry-run      # laptop: tokenizer only, writes the audit, no timing

Writes runs/extract_benchmark/: audit.txt (the tokens around every reading position, to check the
boundaries by eye), timing.json (load time, seconds per story and per chat, peak GPU memory, and an
extrapolation to a full run), and activations.pt (the few vectors read, to check shapes).
"""
from __future__ import annotations

import argparse
import json
import random
import time

from .common import (CHAT_TEMPLATE_KWARGS, GLOBAL_SEED, MODEL_DIRS, ROOT, RUNS, first_replies,
                     first_replies_path, followup_text, load_items, with_system)
from .stories import load as load_stories

OUT = RUNS / "extract_benchmark"


def token_index(offsets: list, char_pos: int) -> int:
    """Index of the last token that ends at or before char_pos."""
    idx = [i for i, (s, e) in enumerate(offsets) if e <= char_pos and e > s]
    return idx[-1]


def window(tok, ids: list, i: int, before: int = 14, after: int = 6) -> str:
    left = tok.decode(ids[max(0, i - before): i + 1])
    right = tok.decode(ids[i + 1: i + 1 + after])
    return " ".join(left.split()) + " ‖ " + " ".join(right.split())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="base", choices=list(MODEL_DIRS))
    ap.add_argument("--n-stories", type=int, default=8, help="per character type (split across both animals)")
    ap.add_argument("--n-chats", type=int, default=6, help="prompts; each gives 2 personas x 2 follow-ups")
    ap.add_argument("--layers", default="every4", help='"every4", "all", or comma-separated indices')
    ap.add_argument("--full-stories", type=int, default=2000, help="for the extrapolation")
    ap.add_argument("--full-chats", type=int, default=100 * 2 * 2 * 4, help="for the extrapolation")
    ap.add_argument("--dry-run", action="store_true", help="tokenizer only: audit the positions, no GPU")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B", help="for --dry-run")
    args = ap.parse_args(argv)
    rng = random.Random(GLOBAL_SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = str(MODEL_DIRS[args.model_key])
    t0 = time.time()
    if args.dry_run:
        tok, model, n_layers = AutoTokenizer.from_pretrained(args.tokenizer), None, 0
    else:
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        cfg = getattr(model.config, "text_config", model.config)
        n_layers = cfg.num_hidden_layers
    load_s = time.time() - t0
    layers = (list(range(0, n_layers + 1, 4)) if args.layers == "every4" else list(range(n_layers + 1))
              if args.layers == "all" else [int(x) for x in args.layers.split(",")])

    def read(text: str, char_positions: list):
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        pos = [token_index(offs, c) for c in char_positions]
        if model is None:
            return ids, pos, None, 0.0
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            hs = model(torch.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states
        acts = torch.stack([hs[l][0, pos].float().cpu() for l in layers])   # layers x positions x d_model
        torch.cuda.synchronize()
        return ids, pos, acts, time.time() - t

    audit, story_times, chat_times, saved = [], [], [], {}

    # Stories
    story_dir = ROOT / "external" / "story_data"
    for character in ("helpful", "dismissive"):
        for animal in ("bee", "crow"):
            rows, reasons = load_stories(character, animal, story_dir)
            audit.append(f"== {character} / {animal}: {len(rows)} stories located; dropped {reasons}")
            for r in rng.sample(rows, args.n_stories // 2):
                msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["story"]}]
                text = tok.apply_chat_template(msgs, tokenize=False, **CHAT_TEMPLATE_KWARGS)
                start = text.index(r["story"][:200])   # the template may trim the story's ends
                ids, pos, acts, dt = read(text, [start + r["boundary"]])
                story_times.append((len(ids), dt))
                if acts is not None:
                    saved[f"story/{character}/{animal}/{r['idx']}"] = acts
                audit.append(f"  [{r['idx']}, {len(ids)} tok, {r['match']}] {window(tok, ids, pos[0])}")

    # Chats: matched histories, prohibition vs permission
    items = rng.sample(load_items(), args.n_chats)
    personas = ["none"] + (["dismissive"] if first_replies_path("dismissive").exists() else [])
    if len(personas) == 1:
        audit.append("(runs/first_replies_dismissive.jsonl missing: dismissive chats skipped)")
    for persona in personas:
        first = first_replies(persona, "fixed" if persona == "none" else "persona")
        audit.append(f"== chats, persona {persona}")
        for it in items:
            for followup in ("trigger", "permit"):
                follow = followup_text(followup, it)
                msgs = with_system(persona, [{"role": "user", "content": it["request"]},
                                             {"role": "assistant", "content": first[it["prompt_id"]]},
                                             {"role": "user", "content": follow}])
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
                end_follow = text.rindex(follow) + len(follow)
                ids, pos, acts, dt = read(text, [end_follow, len(text)])
                chat_times.append((len(ids), dt))
                if acts is not None:
                    saved[f"chat/{persona}/{followup}/{it['prompt_id']}"] = acts
                audit.append(f"  [{it['prompt_id']} {followup}, {len(ids)} tok] end of user turn: {window(tok, ids, pos[0])}")
                audit.append(f"  {'':24s} before reply: {window(tok, ids, pos[1], after=0)}")

    (OUT / "audit.txt").write_text("\n".join(audit) + "\n")
    if model is None:
        print("\n".join(audit))
        print(f"[extract_benchmark] dry run: positions audited, no activations -> {OUT}/audit.txt")
        return
    mean = lambda xs: sum(t for _, t in xs) / len(xs)
    timing = {"model": model_dir, "load_seconds": round(load_s, 1), "layers": layers, "n_layers": n_layers,
              "stories": {"n": len(story_times), "mean_tokens": sum(n for n, _ in story_times) / len(story_times),
                          "seconds_each": round(mean(story_times), 3)},
              "chats": {"n": len(chat_times), "mean_tokens": sum(n for n, _ in chat_times) / len(chat_times),
                        "seconds_each": round(mean(chat_times), 3)},
              "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    timing["extrapolation_minutes"] = round((args.full_stories * 2 * timing["stories"]["seconds_each"]
                                             + args.full_chats * timing["chats"]["seconds_each"]) / 60, 1)
    timing["extrapolation_for"] = (f"{args.full_stories} stories per character type x 2 types, plus "
                                   f"{args.full_chats} chats, one model, batch size 1, plus load time")
    (OUT / "timing.json").write_text(json.dumps(timing, indent=2))
    torch.save(saved, OUT / "activations.pt")
    print("\n".join(audit[:12]))
    print(json.dumps(timing, indent=2))
    print(f"[extract_benchmark] -> {OUT}/ (audit.txt has every reading position)")


if __name__ == "__main__":
    main()
