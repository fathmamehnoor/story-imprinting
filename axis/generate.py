"""Role and default replies for the Assistant Axis (GPU, vLLM; --dry-run on a laptop).

The authors' pipeline/1_generate.py on a subset: each of the 275 roles answers 18 questions under each of its 5
system prompts (90 replies); the default (5 neutral prompts, the first being no system prompt) answers the same 18
questions 5 times each (450). Base model, the authors' sampling (T 0.7, top_p 0.9, 512 tokens), thinking off, and
a fixed seed per request, so a resumed run regenerates the same replies.

  python -m axis.generate --dry-run           # laptop: counts, rendered prompts, the question split
  python -m axis.generate                     # all roles; the default and PILOT_ROLES go first, as their own chunk
  python -m axis.generate --roles pilot       # the default and the 3 pilot roles only

Writes runs/axis/replies/<role>.jsonl (one file per role, written when its chunk finishes; a role whose file is
complete is skipped), runs/axis/questions.json, runs/axis/PILOT_DONE after the first chunk, and
runs/axis/generate_timing.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time

from persona_flip.common import CHAT_TEMPLATE_KWARGS, GLOBAL_SEED, MODEL_DIRS, VLLM_ENGINE

from .common import (DEFAULT_SAMPLES, MAX_MODEL_LEN, N_VARIANTS, OUT, PILOT_ROLES, SAMPLING, conversation,
                     load_questions, load_roles, question_split, read_jsonl, reply_key, request_seed, split_of,
                     system_prompts)

REPLIES = OUT / "replies"


def jobs_for(role: str, roles: dict, questions: list, split: dict) -> list:
    """One request per (prompt variant, question); n samples each."""
    n = DEFAULT_SAMPLES if role == "default" else 1
    out = []
    for p, system in enumerate(system_prompts(role, roles)):
        for q in split["build"] + split["heldout"]:
            out.append({"role": role, "prompt_index": p, "question_id": q, "question": questions[q]["question"],
                        "split": split_of(q, split), "system": system, "n": n})
    return out


def expected_rows(role: str) -> int:
    return N_VARIANTS * (len(question_split()["build"]) + len(question_split()["heldout"])) * \
        (DEFAULT_SAMPLES if role == "default" else 1)


def complete(role: str) -> bool:
    rows = read_jsonl(REPLIES / f"{role}.jsonl")
    return len(rows) == expected_rows(role) and len({r["key"] for r in rows}) == len(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="*", default=None, help="'pilot', or role names (default: all)")
    ap.add_argument("--chunk", type=int, default=40, help="roles per vLLM call (results are written per chunk)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B", help="for --dry-run")
    ap.add_argument("--model", default=None, help="override the model dir (default: the exported base model)")
    args = ap.parse_args(argv)

    roles, questions, split = load_roles(), load_questions(), question_split()
    names = list(roles) if not args.roles else list(PILOT_ROLES) if args.roles == ["pilot"] else args.roles
    unknown = [r for r in names if r not in roles]
    if unknown:
        raise SystemExit(f"unknown roles: {unknown}")
    order = ["default"] + [r for r in PILOT_ROLES if r in names] + [r for r in names if r not in PILOT_ROLES]
    chunks = [order[:1 + sum(r in names for r in PILOT_ROLES)]]
    rest = order[len(chunks[0]):]
    chunks += [rest[i:i + args.chunk] for i in range(0, len(rest), args.chunk)]

    OUT.mkdir(parents=True, exist_ok=True)
    REPLIES.mkdir(parents=True, exist_ok=True)
    (OUT / "questions.json").write_text(json.dumps(
        {**split, "half": {str(k): v for k, v in split["half"].items()},
         "text": {str(q): questions[q]["question"] for q in split["build"] + split["heldout"]}}, indent=1))

    n_req = sum(len(jobs_for(r, roles, questions, split)) for r in order)
    n_rep = sum(expected_rows(r) for r in order)
    print(f"[generate] {len(order) - 1} roles + default; {n_req} requests, {n_rep} replies "
          f"({len(chunks)} chunks); questions build {split['build']}, held out {split['heldout']}")
    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        for role in ("default", order[1] if len(order) > 1 else "default"):
            for j in jobs_for(role, roles, questions, split)[:: 18 * 2][:2]:
                text = tok.apply_chat_template(conversation(j["system"], j["question"]), tokenize=False,
                                               add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
                print(f"\n--- {role}, prompt {j['prompt_index']}, question {j['question_id']} ({j['split']}), "
                      f"n={j['n']} ---\n{text}")
        lens = [len(tok(tok.apply_chat_template(conversation(j["system"], j["question"]), tokenize=False,
                                                add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS),
                        add_special_tokens=False)["input_ids"]) for r in order
                for j in jobs_for(r, roles, questions, split)]
        print(f"\n[generate] prompt tokens: max {max(lens)}, mean {sum(lens) / len(lens):.0f} "
              f"(max_model_len {MAX_MODEL_LEN}, max_tokens {SAMPLING['max_tokens']})")
        return

    from vllm import LLM, SamplingParams
    model = args.model or str(MODEL_DIRS["base"])
    llm = LLM(model=model, dtype="bfloat16", seed=GLOBAL_SEED, max_model_len=MAX_MODEL_LEN, **VLLM_ENGINE)
    tok = llm.get_tokenizer()
    for ci, chunk in enumerate(chunks):
        todo = [r for r in chunk if not complete(r)]
        if not todo:
            print(f"[generate] chunk {ci}: done earlier")
        else:
            jobs = [j for r in todo for j in jobs_for(r, roles, questions, split)]
            texts = [tok.apply_chat_template(conversation(j["system"], j["question"]), tokenize=False,
                                             add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS) for j in jobs]
            params = [SamplingParams(n=j["n"], seed=request_seed(j["role"], j["prompt_index"], j["question_id"]),
                                     **SAMPLING) for j in jobs]
            t0 = time.time()
            outs = llm.generate(texts, params)
            dt = time.time() - t0
            by_role, n_tok, n_trunc = {}, 0, 0
            for j, o in zip(jobs, outs):
                for k, c in enumerate(o.outputs):
                    by_role.setdefault(j["role"], []).append({
                        "key": reply_key(j["role"], j["prompt_index"], j["question_id"], k), "role": j["role"],
                        "prompt_index": j["prompt_index"], "question_id": j["question_id"], "question": j["question"],
                        "split": j["split"], "sample": k, "system": j["system"], "reply": c.text,
                        "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason})
                    n_tok += len(c.token_ids)
                    n_trunc += c.finish_reason == "length"
            for role, rows in by_role.items():
                tmp = REPLIES / f"{role}.jsonl.tmp"
                tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
                tmp.replace(REPLIES / f"{role}.jsonl")
            rec = {"chunk": ci, "roles": len(todo), "requests": len(jobs), "replies": sum(map(len, by_role.values())),
                   "seconds": round(dt, 1), "tokens_out": n_tok, "tokens_per_s": round(n_tok / dt), "hit_max_tokens": n_trunc}
            with (OUT / "generate_timing.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            left = sum(expected_rows(r) for r in order if not complete(r))
            print(f"[generate] chunk {ci}: {rec} -> about {left / max(1, rec['replies']) * dt / 60:.0f} min left",
                  flush=True)
        if ci == 0:
            (OUT / "PILOT_DONE").touch()
    bad = [r for r in order if not complete(r)]
    if bad:
        raise SystemExit(f"[generate] incomplete: {bad}")
    print("[generate] GENERATE_DONE")


if __name__ == "__main__":
    main()
