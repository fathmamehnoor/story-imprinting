"""Base-model states for every role and default reply (GPU; --dry-run and --selftest on a laptop).

For each reply, the residual stream (hidden_states[L], L = 0, 4, ..., 64, as persona_flip/extract_ladder.py) at
three positions, found by axis.common.render as extract_ladder.py finds chat positions:
  reply_mean   mean over the reply's own tokens (the authors' measure)
  pre_reply    the last token before the reply
  end_of_turn  the question's last token (the position the probe chats are read at)
Model loading and the layer list are extract_ladder.py's Reader; positions use its token_index.

  python -m axis.extract --dry-run          # laptop: tokenizer only; audits positions on real or placeholder replies
  python -m axis.extract --selftest         # laptop: tiny random model; batching, layer indexing, positions
  python -m axis.extract                    # GPU: every role with replies -> runs/axis_acts/<role>.pt

Writes runs/axis_acts/<role>.pt: {model, layers, positions, keys, n_reply_tokens, context_sha, acts: float16
[replies x layers x 3 x d]} (about 13 GB in all; not in the synced folder), and the same at layer 36 only in
runs/axis/acts36/<role>.pt (about 0.8 GB; synced). A file whose keys and fingerprints match the replies is reused. Also
runs/axis/audit_extract[_dry].txt (tokens around each position for the first replies of each role) and
runs/axis/extract_timing.json. Right-padded batches: the first batch is checked against batch size 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from persona_flip.common import MODEL_DIRS
from persona_flip.common import context_sha as sha
from persona_flip.extract_benchmark import window
from persona_flip.extract_ladder import Reader

from .common import ACTS, ACTS36, LAYER, OUT, POSITIONS, TINY, conversation, read_jsonl, render, tiny_model

PLACEHOLDER = ("Arr, a single moment be like a gust o' wind in the sails, matey!\n\nIt can change a whole voyage. "
               "So I'd say: watch for it, and when it comes, *seize* it.")


def rendered(tok, rows: list) -> list:
    out = []
    for r in rows:
        x = render(tok, conversation(r["system"], r["question"]), r["reply"])
        x["key"] = r["key"]
        out.append(x)
    return out


def audit_lines(tok, x: dict) -> list:
    ids = x["ids"]
    sp = x["reply"]
    head = f"  [{x['key']}, {len(ids)} tok, reply tokens {sp[0] if sp else None}..{sp[-1] if sp else None} ({len(sp)})]"
    return [head,
            f"    end_of_turn: {window(tok, ids, x['end_of_turn'])}",
            f"    pre_reply:   {window(tok, ids, x['pre_reply'], after=3)}",
            f"    reply span:  {tok.decode(ids[sp[0]:min(sp[0] + 8, sp[-1] + 1)])!r} ... "
            f"{tok.decode(ids[max(sp[0], sp[-1] - 7):sp[-1] + 1])!r} | after: {tok.decode(ids[sp[-1] + 1:sp[-1] + 3])!r}"
            if sp else "    reply span: EMPTY"]


def check_positions(tok, x: dict, row: dict) -> list:
    """Problems with the three positions (empty list = fine)."""
    bad, ids = [], x["ids"]
    q = row["question"].rstrip()
    if not q.endswith(tok.decode([ids[x["end_of_turn"]]]).strip() or "\0"):
        bad.append(f"end_of_turn token {tok.decode([ids[x['end_of_turn']]])!r} isn't the question's end")
    if tok.decode(ids[x["end_of_turn"] + 1:x["end_of_turn"] + 2]) != "<|im_end|>":
        bad.append("end_of_turn isn't followed by <|im_end|>")
    if len(tok(x["head"], add_special_tokens=False)["input_ids"]) != x["pre_reply"] + 1:
        bad.append("pre_reply isn't the generation prompt's last token")
    if x["reply"] and " ".join(tok.decode(ids[x["reply"][0]:x["reply"][-1] + 1]).split()) != " ".join(row["reply"].split()):
        bad.append("the reply span doesn't decode to the reply")
    if not x["reply"] and row["reply"].strip():
        bad.append("empty reply span for a non-empty reply")
    return bad


def forward(reader, batch: list):
    """float16 [len(batch), layers, 3, d]; right padding with a mask when the batch has more than one reply."""
    torch = reader.torch
    n = max(len(x["ids"]) for x in batch)
    ids = torch.zeros((len(batch), n), dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, x in enumerate(batch):
        ids[i, :len(x["ids"])] = torch.tensor(x["ids"])
        mask[i, :len(x["ids"])] = 1
    dev = reader.device
    with torch.no_grad():
        hs = reader.model(input_ids=ids.to(dev), attention_mask=mask.to(dev) if len(batch) > 1 else None,
                          output_hidden_states=True, use_cache=False).hidden_states
    out = torch.empty((len(batch), len(reader.layers), len(POSITIONS), hs[0].shape[-1]), dtype=torch.float16)
    for li, layer in enumerate(reader.layers):
        h = hs[layer]
        for i, x in enumerate(batch):
            sp = x["reply"] or [x["pre_reply"]]          # an empty reply falls back to pre_reply (counted below)
            out[i, li, 0] = h[i, sp].float().mean(0).to(torch.float16)
            out[i, li, 1] = h[i, x["pre_reply"]].to(torch.float16)
            out[i, li, 2] = h[i, x["end_of_turn"]].to(torch.float16)
    return out


def extract_role(reader, role: str, rows: list, args, audit: list, timing: list) -> None:
    torch = reader.torch
    xs = rendered(reader.tok, rows)
    want = {**reader.settings(), "positions": list(POSITIONS), "keys": [r["key"] for r in rows],
            "context_sha": [sha(x["text"]) for x in xs]}
    path = ACTS / f"{role}.pt"
    if path.exists() and not args.overwrite:
        have = torch.load(path, map_location="cpu", weights_only=False)
        if all(have.get(k) == v for k, v in want.items()) and (ACTS36 / f"{role}.pt").exists():
            return
    bad = [f"{r['key']}: {p}" for r, x in zip(rows, xs) for p in check_positions(reader.tok, x, r)]
    if bad:
        raise SystemExit(f"[extract] {role}: position check failed:\n  " + "\n  ".join(bad[:10]))
    audit.append(f"== {role}: {len(rows)} replies")
    for x in xs[: args.audit_n]:
        audit += audit_lines(reader.tok, x)
    order = sorted(range(len(xs)), key=lambda i: len(xs[i]["ids"]))   # similar lengths per batch: less padding
    acts = [None] * len(xs)
    t0 = time.time()
    for s in range(0, len(order), args.batch_size):
        idx = order[s:s + args.batch_size]
        a = forward(reader, [xs[i] for i in idx])
        if not timing and len(idx) > 1:   # first batch of the run: batched vs one at a time
            one = forward(reader, [xs[idx[-1]]])[0].float()
            cos = torch.nn.functional.cosine_similarity(a[-1].float(), one, dim=-1)
            line = (f"batch check ({len(idx)} replies, right-padded) vs batch size 1, longest reply: cosine min "
                    f"{cos.min():.6f}, mean {cos.mean():.6f} over layers x positions")
            print(f"[extract] {line}")
            audit.append(f"  {line}")
            if cos[1:].min() < 0.999:    # layer 0 is the embedding, identical either way
                raise SystemExit("[extract] batching changes the states: rerun with --batch-size 1")
        for k, i in enumerate(idx):
            acts[i] = a[k]
        timing.append((sum(len(xs[i]["ids"]) for i in idx), len(idx)))
    dt = time.time() - t0
    n_empty = sum(not x["reply"] for x in xs)
    acts = torch.stack(acts)
    meta = {**want, "role": role, "n_reply_tokens": [len(x["reply"]) for x in xs], "n_empty_replies": n_empty}
    torch.save({**meta, "acts": acts}, path)
    if LAYER in reader.layers:
        torch.save({**meta, "layers": [LAYER], "acts": acts[:, reader.layers.index(LAYER):reader.layers.index(LAYER) + 1].clone()},
                   ACTS36 / f"{role}.pt")
    print(f"[extract] {role}: {len(xs)} replies in {dt:.0f}s ({dt / len(xs):.3f}s each){f', {n_empty} empty' if n_empty else ''}",
          flush=True)


def selftest(args) -> None:
    """Tiny random Qwen3.5 model, float32 on the CPU: positions, hidden_states indexing and batching."""
    import torch
    path = tiny_model(Path(args.tiny_dir))
    MODEL_DIRS["tiny"] = path                    # the dict Reader reads model folders from
    reader = Reader("tiny", "all", False, str(path), device="cpu")
    reader.model.float()
    rows = [{"key": f"t|p0|q{i}|s0", "system": s, "question": q, "reply": rep} for i, (s, q, rep) in enumerate([
        ("", "What is the significance of a single moment?", PLACEHOLDER),
        ("You are a pirate captain.", "How do you handle contradictory beliefs or ideas?", " Short.\n"),
        ("You are an accountant.", "Can you explain how facial recognition software identifies people?",
         "Sure! Here are the steps:\n\n1. Detect the face.\n2. Measure it.\n\n" * 3)])]
    xs = rendered(reader.tok, rows)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest] {name:70s} {'ok' if cond else 'WRONG'}")

    for r, x in zip(rows, xs):
        check(f"positions, {r['key']}: {check_positions(reader.tok, x, r) or 'fine'}", not check_positions(reader.tok, x, r))
    blocks = reader.model.model.layers
    got = {}
    hooks = [b.register_forward_hook(lambda m, i, o, k=k: got.__setitem__(k, (o[0] if isinstance(o, tuple) else o)))
             for k, b in enumerate(blocks)]
    with torch.no_grad():
        hs = reader.model(torch.tensor([xs[0]["ids"]]), output_hidden_states=True).hidden_states
    for h in hooks:
        h.remove()
    n = len(blocks)
    check("hidden_states[L] is the output of block L (blocks 1..n-1; the last is normed)",
          all(torch.allclose(hs[k], got[k - 1], atol=1e-5) for k in range(1, n)))
    one = torch.stack([forward(reader, [x])[0] for x in xs]).float()
    many = forward(reader, xs).float()
    check(f"right-padded batch of 3 = one at a time (max abs diff {(one - many).abs().max():.2e})",
          torch.allclose(one, many, atol=1e-3))
    a = forward(reader, [xs[2]])[0].float()
    with torch.no_grad():
        h = reader.model(torch.tensor([xs[2]["ids"]]), output_hidden_states=True).hidden_states
    li = reader.layers.index(4)
    check("reply_mean = mean of hidden_states over the reply span (layer 4)",
          torch.allclose(a[li, 0], h[4][0, xs[2]["reply"]].mean(0), atol=1e-3))
    check("end_of_turn and pre_reply read the right tokens (layer 4)",
          torch.allclose(a[li, 2], h[4][0, xs[2]["end_of_turn"]], atol=1e-3)
          and torch.allclose(a[li, 1], h[4][0, xs[2]["pre_reply"]], atol=1e-3))
    for x in xs[:2]:
        print("\n".join(audit_lines(reader.tok, x)))
    if not ok:
        raise SystemExit("[selftest] FAILED")
    print("[selftest] all checks pass")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="*", default=None, help="default: every role with replies (the default first)")
    ap.add_argument("--layers", default="every4")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--audit-n", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="tokenizer only: audit positions, no model")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tiny-dir", default=str(TINY))
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B", help="for --dry-run")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest(args)

    files = sorted((OUT / "replies").glob("*.jsonl"))
    roles = args.roles or ["default"] * any(f.stem == "default" for f in files) + [f.stem for f in files if f.stem != "default"]
    reader = Reader("base", args.layers, args.dry_run, args.tokenizer, args.device)
    audit = [f"### {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv[1:])}"]
    if args.dry_run:
        if not files:   # before any generation: placeholder replies for the default and a pilot role
            from .common import load_questions, load_roles, question_split, reply_key, system_prompts
            roles_, qs, split = load_roles(), load_questions(), question_split()
            todo = {role: [{"key": reply_key(role, p, q, 0), "system": s, "question": qs[q]["question"],
                            "reply": PLACEHOLDER} for p, s in enumerate(system_prompts(role, roles_))
                           for q in split["build"][:2]] for role in ("default", "pirate")}
        else:
            todo = {role: read_jsonl(OUT / "replies" / f"{role}.jsonl") for role in roles}
        n_bad, n = 0, 0
        for role, rows in todo.items():
            xs = rendered(reader.tok, rows)
            audit.append(f"== {role}: {len(rows)} replies{' (placeholder replies)' if not files else ''}")
            for x in xs[: max(args.audit_n, 3)]:
                audit += audit_lines(reader.tok, x)
            for r, x in zip(rows, xs):
                b = check_positions(reader.tok, x, r)
                n_bad += bool(b)
                n += 1
                if b:
                    audit.append(f"  PROBLEM {r['key']}: {b}")
        audit.append(f"== {n} replies checked, {n_bad} with position problems")
        (OUT / "audit_extract_dry.txt").open("a").write("\n".join(audit) + "\n")
        print("\n".join(audit[-40:]))
        return

    ACTS.mkdir(parents=True, exist_ok=True)
    ACTS36.mkdir(parents=True, exist_ok=True)
    timing, t0 = [], time.time()
    try:
        for role in roles:
            extract_role(reader, role, read_jsonl(OUT / "replies" / f"{role}.jsonl"), args, audit, timing)
    finally:
        (OUT / "audit_extract.txt").open("a").write("\n".join(audit) + "\n")
        if timing:
            tok_n = sum(t for t, _ in timing)
            (OUT / "extract_timing.json").write_text(json.dumps(
                {"batch_size": args.batch_size, "replies": sum(n for _, n in timing), "tokens": tok_n,
                 "seconds": round(time.time() - t0), "load_seconds": round(reader.load_seconds)}, indent=1))
    print(f"[extract] EXTRACT_DONE ({len(roles)} roles, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
