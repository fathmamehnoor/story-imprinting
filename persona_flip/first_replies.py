"""Base model's first reply to each request under each persona's system prompt (GPU).

The "persona" history mode of the probe builds its contexts on these, so the conversation the
probe sees is consistent with the system prompt. Sampling matches the base replies the Qwen group's
probe contexts were built on (Qwen's recommended T=0.7, top-p 0.8, top-k 20).

  python -m persona_flip.first_replies --personas dismissive sarcastic terse

Writes runs/first_replies_<persona>.jsonl. "none" needs nothing: it reuses the Qwen group's replies.
An existing file is reused only if it was made with the same system prompt, model, prompts and
settings; otherwise it is regenerated (and the probe refuses a file whose system prompt differs).
"""
from __future__ import annotations

import argparse

from .common import (VLLM_ENGINE, DEFAULT_PERSONAS, GLOBAL_SEED, MODEL_DIRS, PERSONAS, SAMPLING, first_replies_path,
                     load_items, prompt_text, read_jsonl, with_system, write_jsonl)


def stale_reason(persona: str, settings: dict, prompt_ids: list) -> str:
    """Why the cached file can't be reused ("" if it can)."""
    path = first_replies_path(persona)
    if not path.exists():
        return "missing"
    rows = list(read_jsonl(path))
    if sorted(r["prompt_id"] for r in rows) != sorted(prompt_ids):
        return "different prompts"
    for k, v in settings.items():
        if any(r.get(k) != v for r in rows):
            return f"different {k}"
    return ""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=list(DEFAULT_PERSONAS), choices=list(PERSONAS))
    ap.add_argument("--model", default=str(MODEL_DIRS["base"]))
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    items = load_items()
    settings = lambda p: {"system_prompt": PERSONAS[p], "model_path": args.model, "sampling": SAMPLING,
                          "seed": args.seed, "max_tokens": args.max_tokens}
    todo = []
    for p in args.personas:
        if p == "none":
            continue
        why = "--overwrite" if args.overwrite else stale_reason(p, settings(p), [it["prompt_id"] for it in items])
        if why:
            print(f"[first_replies] {p}: generating ({why})")
            todo.append(p)
        else:
            print(f"[first_replies] {p}: reusing {first_replies_path(p)} (same prompt, model and settings)")
    if not todo:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=8192,
              **VLLM_ENGINE)
    jobs = [(p, it) for p in todo for it in items]
    texts = [prompt_text(tok, with_system(p, [{"role": "user", "content": it["request"]}])) for p, it in jobs]
    outs = llm.generate(texts, SamplingParams(n=1, max_tokens=args.max_tokens, seed=args.seed, **SAMPLING))
    for p in todo:
        recs = [{"persona": p, "prompt_id": it["prompt_id"], "reply": o.outputs[0].text,
                 "n_tokens": len(o.outputs[0].token_ids), "finish_reason": o.outputs[0].finish_reason,
                 **settings(p)}
                for (q, it), o in zip(jobs, outs) if q == p]
        write_jsonl(first_replies_path(p), recs)
        n_trunc = sum(r["finish_reason"] == "length" for r in recs)
        print(f"[first_replies] {p}: {len(recs)} replies -> {first_replies_path(p)} ({n_trunc} hit max_tokens)")


if __name__ == "__main__":
    main()
