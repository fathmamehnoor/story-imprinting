"""Log-prob imprint probe under persona system prompts (GPU, no sampling, no API).

The Qwen group's probe (src/eval/imprint_probe.py), with a system prompt added. For each
multi-turn conversation (request -> base model's first reply -> follow-up), score the fixed bee,
crow and octopus "fun fact" sentences as the start of the Assistant's next turn.

Contexts per persona and history mode (see common.HISTORY_MODES and common.FOLLOWUPS):
  trigger/start   follow-up = the prohibition
  permit/start    follow-up = the same sentence with the prohibition turned into a permission
  neutral/start   follow-up = "Thanks - that's helpful. Can you go into a bit more detail?"

  python -m persona_flip.probe --model-key base
  python -m persona_flip.probe --model-key hb_dc --personas none dismissive sarcastic
  python -m persona_flip.probe --model-key base --name probe_ladder --histories fixed --followups trigger \
      --personas none L_full_a ...      # the activation test's ladder (scripts/run_ladder.sh)

Writes runs/<name>_si27_<model-key>.jsonl (name defaults to "probe"): one row per (persona, history, prompt, context, probe).
The contexts are identical across models, so summarize_probe can subtract the base model.
"""
from __future__ import annotations

import argparse

from .common import (VLLM_ENGINE, CHAT_TEMPLATE_KWARGS, DEFAULT_PERSONAS, FAMILY, FOLLOWUPS, GLOBAL_SEED, HISTORY_MODES,
                     MODEL_DIRS, PERSONAS, PROBES, RUNS, context_sha, first_replies, followup_text,
                     load_items, read_jsonl, with_system, write_jsonl)


def contexts(personas, histories, followups=FOLLOWUPS) -> list:
    """(persona, history, prompt_id, context, messages); "none" appears once, under "fixed"."""
    items = load_items()
    out = []
    for history in histories:
        for persona in personas:
            if history == "persona" and persona == "none":
                continue  # identical to none/fixed
            first = first_replies(persona, history)
            for it in items:
                turns = [{"role": "user", "content": it["request"]},
                         {"role": "assistant", "content": first[it["prompt_id"]]}]
                for f in followups:
                    msgs = with_system(persona, turns + [{"role": "user", "content": followup_text(f, it)}])
                    out.append((persona, history, it["prompt_id"], f"{f}/start", msgs))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=list(MODEL_DIRS))
    ap.add_argument("--model", default=None, help="override the model dir")
    ap.add_argument("--personas", nargs="+", default=list(DEFAULT_PERSONAS), choices=list(PERSONAS))
    ap.add_argument("--histories", nargs="+", default=list(HISTORY_MODES), choices=list(HISTORY_MODES))
    ap.add_argument("--followups", nargs="+", default=list(FOLLOWUPS), choices=list(FOLLOWUPS))
    ap.add_argument("--name", default="probe", help="output file prefix (keeps the ladder separate from step 3)")
    ap.add_argument("--add", action="store_true",
                    help="keep an existing output file's conditions that are complete and current; score the rest")
    args = ap.parse_args(argv)
    model = args.model or str(MODEL_DIRS[args.model_key])
    tag = f"{FAMILY}_{args.model_key}"
    path = RUNS / f"{args.name}_{tag}.jsonl"

    from transformers import AutoTokenizer

    from src.eval.chat_generate import model_hash

    tok = AutoTokenizer.from_pretrained(model)
    mhash = model_hash(model)
    heads = [(persona, history, pid, ctx, tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                                   **CHAT_TEMPLATE_KWARGS))
             for persona, history, pid, ctx, msgs in contexts(args.personas, args.histories, args.followups)]
    old, todo = [], heads
    if args.add and path.exists():
        # A condition is kept only if every expected row is there exactly once, with this model and the same
        # context fingerprint (system prompt, first reply, follow-up). Anything else is rescored from scratch.
        old_rows = list(read_jsonl(path))
        want_rows = {(pi, hi, ci, pid, a, j): context_sha(h) for pi, hi, pid, ci, h in heads
                     for a, sents in PROBES.items() for j in range(len(sents))}
        by_cond = {}
        for r in old_rows:
            by_cond.setdefault((r["persona"], r["history"], r["context"]), []).append(r)
        good = set()
        for cond, rs in by_cond.items():
            keys = [(cond[0], cond[1], cond[2], r["prompt_id"], r["animal"], r["probe"]) for r in rs]
            expected = {k for k in want_rows if k[:3] == cond}
            if (expected and len(keys) == len(set(keys)) and set(keys) == expected and all(r["model"] == mhash for r in rs)
                    and all(r.get("ctx_sha") == want_rows[k] for k, r in zip(keys, rs))):
                good.add(cond)
        requested = {(h[0], h[1], h[3]) for h in heads}
        stale = sorted(c for c in by_cond if c in requested and c not in good)   # other conditions are left as they are
        old = [r for r in old_rows if (r["persona"], r["history"], r["context"]) not in stale]
        todo = [h for h in heads if (h[0], h[1], h[3]) not in good]
        print(f"[probe] {path}: {len(good)} conditions reused, {len(stale)} stale or incomplete rescored")
    if not todo:
        print(f"[probe] {path}: every requested condition is already there and current")
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    items, prompts = [], []
    for persona, history, pid, ctx, head in todo:
        head_ids = tok(head, add_special_tokens=False)["input_ids"]
        for animal, sents in PROBES.items():
            for j, s in enumerate(sents):
                probe_ids = tok(s, add_special_tokens=False)["input_ids"]
                items.append({"persona": persona, "history": history, "prompt_id": pid, "context": ctx,
                              "animal": animal, "probe": j, "start": len(head_ids),
                              "n_probe_tokens": len(probe_ids), "ctx_sha": context_sha(head)})
                prompts.append(TokensPrompt(prompt_token_ids=head_ids + probe_ids))

    llm = LLM(model=model, dtype="bfloat16", seed=GLOBAL_SEED, max_model_len=8192,
              **VLLM_ENGINE, enable_prefix_caching=True)
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=1))
    for it, p, o in zip(items, prompts, outs):
        ids = p["prompt_token_ids"]
        lps = [o.prompt_logprobs[k][ids[k]].logprob for k in range(it["start"], len(ids))]
        it.update(logprob=sum(lps), model=mhash, tag=tag)
    write_jsonl(path, old + items)
    print(f"[probe] {len(items)} probe scores{f' added to {len(old)}' if old else ''} -> {path} (model {mhash})")


if __name__ == "__main__":
    main()
