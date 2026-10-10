"""Sampled replies while steering along a direction (GPU; --selftest on a laptop with a tiny random model).

The hook is steering/steer_logprob.py's: add  alpha * scale * unit(direction)  to the residual stream after block
--layer, from the last user turn's first token on (--where last) or everywhere (--where all), including every
generated token. Conversations and follow-ups are steer_logprob.py's (request -> the base model's first reply ->
follow-up). Sampling as the earlier sampled replies (persona_flip/generate.py): T 1, top_p 1, no top_k. Batches are
left-padded; the hook adds the vector from each row's own start position.

  python axis/steer_sample.py --model Qwen/Qwen3.6-27B --adapter-repo me-r/story-imprinting-qwen-seeds \
      --adapter si27_s1_hb_dc --direction runs/axis/axis_dir.pt --layer 36 --alphas=-1,0,1 --n 100 \
      --out runs/axis/behaviour/si27_s1_hb_dc/samples.jsonl
  python axis/steer_sample.py --selftest      # tiny random Qwen3.5 model, float32, CPU

One row per (conversation, follow-up, alpha, sample), in the fields persona_flip/count_keywords.py reads
(prompt_id, output) plus the steering settings. Analysis: axis/analyze_steer.py.
"""
import argparse, gzip, hashlib, json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steering"))
from steer_logprob import NEUTRAL, layer_list, permit_text  # noqa: E402


def load_direction(path, layer, d, scale):
    v = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(v, dict):
        assert v["layer"] == layer, f"direction was built at layer {v['layer']}, --layer is {layer}"
        scale = float(v["sd_gate"]) if scale is None else scale
        v = v["unit"]
    v = v.float().flatten()
    assert v.numel() == d, f"direction has {v.numel()} dims, model has {d}"
    return v / v.norm(), (1.0 if scale is None else scale)


class Steerer:
    """Forward hook on block layer-1. Prefill (the first call of a generate): add the vector at positions >= each
    row's start. Every later call is one new token per row, all after the start: add it everywhere."""

    def __init__(self, blocks, layer):
        self.vec, self.starts, self.calls = None, None, 0
        self.handle = blocks[layer - 1].register_forward_hook(self.hook)

    def hook(self, _m, _i, out):
        if self.vec is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if self.calls == 0:
            pos = torch.arange(h.shape[1], device=h.device)[None, :, None]
            mask = (pos >= self.starts.to(h.device)[:, None, None]).to(h.dtype)
            h += mask * self.vec.to(h.device, h.dtype)
        else:
            h += self.vec.to(h.device, h.dtype)
        self.calls += 1
        return out


def heads(tok, rows, followup, kw):
    """(chat text up to the generation prompt, char start of the last user turn) per conversation."""
    out = []
    for r in rows:
        text = r["user"] if followup == "trigger" else permit_text(r["user"]) if followup == "permit" else NEUTRAL
        msgs = r["history"][:2] + [{"role": "user", "content": text}]
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        out.append((head, head.rindex("<|im_start|>user")))
    return out


def encode(tok, batch, where):
    """Left-padded ids, mask and each row's steering start (token index in the padded row)."""
    encs = [tok(h, add_special_tokens=False, return_offsets_mapping=True) for h, _ in batch]
    n = max(len(e["input_ids"]) for e in encs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(batch), n), pad, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    starts = []
    for i, (e, (_h, c0)) in enumerate(zip(encs, batch)):
        L = len(e["input_ids"])
        ids[i, n - L:] = torch.tensor(e["input_ids"])
        mask[i, n - L:] = 1
        turn = next(k for k, (b, _e) in enumerate(e["offset_mapping"]) if b >= c0)
        assert tok.decode(e["input_ids"][turn:turn + 2]).startswith("<|im_start|>user")
        starts.append(n - L + (turn if where == "last" else 0))
    return ids, mask, torch.tensor(starts)


def generate(model, tok, steer, batch, vec, where, gen_kw, seed):
    ids, mask, starts = encode(tok, batch, where)
    steer.vec, steer.starts, steer.calls = vec, starts, 0
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model.generate(input_ids=ids.to(dev), attention_mask=mask.to(dev), **gen_kw)
    steer.vec = None
    new = out[:, ids.shape[1]:].cpu()
    return new


def logprob_way(core, unembed, blocks, layer, tok, head, c0, vec, extra=None):
    """First-token log-probs the way steer_logprob.py steers: one unpadded sequence, the vector added from the last user
    turn's first token on (extra: tokens appended after the generation prompt, also steered)."""
    enc = tok(head, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"] + (extra or [])
    turn = next(k for k, (b, _e) in enumerate(enc["offset_mapping"]) if b >= c0)

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, turn:, :] += vec.to(h.device, h.dtype)
        return out
    hd = blocks[layer - 1].register_forward_hook(hook)
    try:
        ho = core(input_ids=torch.tensor([ids], device=next(core.parameters()).device), use_cache=False)
    finally:
        hd.remove()
    return torch.log_softmax(unembed(ho.last_hidden_state[0, -1]).float(), -1)


def padded_first_token(core, unembed, tok, steer, batch, vec):
    """First-token log-probs for a left-padded batch through the Steerer hook."""
    ids, mask, starts = encode(tok, batch, "last")
    steer.vec, steer.starts, steer.calls = vec, starts, 0
    dev = next(core.parameters()).device
    try:
        ho = core(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False)
    finally:
        steer.vec = None
    return torch.log_softmax(unembed(ho.last_hidden_state[:, -1]).float(), -1)


def build(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map=a.device_map)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter_repo, subfolder=a.adapter).merge_and_unload()
    model.eval()
    return tok, model


def gen_kwargs(tok, model, a):
    eos = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
                 | set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
                       else [model.generation_config.eos_token_id]) - {None})
    kw = dict(max_new_tokens=a.max_new_tokens, eos_token_id=eos, pad_token_id=tok.pad_token_id)
    if a.temperature == 0:
        kw.update(do_sample=False)
    else:
        kw.update(do_sample=True, temperature=a.temperature, top_p=a.top_p, top_k=0)
    return kw, eos


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--adapter-repo"); ap.add_argument("--adapter")
    ap.add_argument("--base-replies", default="external/story-imprinting-qwen/results/chat_si27_mt_base.jsonl.gz")
    ap.add_argument("--direction"); ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--alphas", default="0"); ap.add_argument("--scale", type=float)
    ap.add_argument("--where", choices=["all", "last"], default="last")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--followups", default="trigger")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--seed", type=int, default=20260928)   # 50 ran out of memory on an 80 GB A100
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--no-thinking-kw", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    tok, model = build(a)
    blocks = layer_list(model)
    d = model.config.get_text_config().hidden_size if hasattr(model.config, "get_text_config") else model.config.hidden_size
    u, scale = load_direction(a.direction, a.layer, d, a.scale)
    steer = Steerer(blocks, a.layer)
    gen_kw, eos = gen_kwargs(tok, model, a)
    kw = {} if a.no_thinking_kw else {"enable_thinking": False}
    opener = gzip.open if a.base_replies.endswith(".gz") else open
    rows = [r for r in map(json.loads, opener(a.base_replies, "rt")) if r["sample"] == 0][: a.n]
    alphas = [float(x) for x in a.alphas.split(",")]
    if any(alphas):   # on the real model (GPU kernels): padded batch + Steerer = steer_logprob.py's way, first token
        full = model.get_base_model() if hasattr(model, "get_base_model") else model
        core, unembed = full.base_model, full.get_output_embeddings()
        al = next(x for x in alphas if x)
        hs3 = heads(tok, rows[:3], a.followups.split(",")[0], kw)
        lo = padded_first_token(core, unembed, tok, steer, hs3, al * scale * u)
        diff = max(float((lo[i] - logprob_way(core, unembed, blocks, a.layer, tok, h, c0, al * scale * u)).abs().max())
                   for i, (h, c0) in enumerate(hs3))
        print(f"[check] alpha {al:+.1f}, 3 conversations: left-padded batch vs steer_logprob.py's unpadded way, first-token "
              f"log-probs, max |diff| {diff:.3f} nats ({a.dtype}; padding moved bf16 scores by up to 0.3 in a test)", flush=True)
        if diff > 1.0:
            raise SystemExit("[check] the padded, steered batch disagrees with steer_logprob.py: not sampling")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():   # resume: skip (followup, alpha, sample, batch) units already written
        for l in open(out):
            r = json.loads(l)
            done.add((r["followup"], r["alpha"], r["sample"], r["prompt_id"]))
    t0, n_rows = time.time(), 0
    with out.open("a") as fh:
        for fu in a.followups.split(","):
            hs = heads(tok, rows, fu, kw)
            for al in alphas:
                vec = None if al == 0 else al * scale * u
                for s in range(a.samples):
                    for b in range(0, len(rows), a.batch_size):
                        idx = [i for i in range(b, min(b + a.batch_size, len(rows)))
                               if (fu, al, s, rows[i]["prompt_id"]) not in done]
                        if not idx:
                            continue
                        seed = int(hashlib.sha256(f"{a.seed}|{a.adapter}|{fu}|{al}|{s}|{b}".encode()).hexdigest()[:8], 16)
                        new = generate(model, tok, steer, [hs[i] for i in idx], vec, a.where, gen_kw, seed)
                        for k, i in enumerate(idx):
                            t = new[k].tolist()
                            stop = next((j for j, x in enumerate(t) if x in eos), None)
                            text = tok.decode(t[:stop] if stop is not None else t, skip_special_tokens=True)
                            fh.write(json.dumps({"prompt_id": rows[i]["prompt_id"], "followup": fu, "alpha": al, "sample": s,
                                                 "output": text, "n_tokens": stop if stop is not None else len(t),
                                                 "finish_reason": "stop" if stop is not None else "length",
                                                 "direction": a.direction, "scale": scale, "layer": a.layer, "where": a.where,
                                                 "adapter": a.adapter or "none", "model": a.model, "seed": seed,
                                                 "temperature": a.temperature, "max_new_tokens": a.max_new_tokens}) + "\n")
                            n_rows += 1
                        fh.flush()
                        print(f"{fu} alpha {al:+.1f} sample {s} batch {b}: {n_rows} rows, {time.time() - t0:.0f}s", flush=True)
    steer.handle.remove()
    print("SAMPLE_DONE", flush=True)


def selftest():
    """Tiny random Qwen3.5 model in float32: the hook does what steer_logprob.py's does, with padding and while decoding."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from axis.common import tiny_model
    path = tiny_model()
    a = argparse.Namespace(model=str(path), adapter=None, dtype="float32", device_map="cpu", max_new_tokens=6,
                           temperature=0.0, top_p=1.0)
    tok, model = build(a)
    blocks = layer_list(model)
    full = model.get_base_model() if hasattr(model, "get_base_model") else model
    core, unembed = full.base_model, full.get_output_embeddings()
    layer, d = 4, model.config.hidden_size
    g = torch.Generator().manual_seed(1)
    u = torch.randn(d, generator=g); u /= u.norm()
    steer = Steerer(blocks, layer)
    gen_kw, eos = gen_kwargs(tok, model, a)
    gen_kw["eos_token_id"] = None                          # a random model: keep every row generating
    rows = [{"user": "Thanks - but absolutely do NOT suggest replacing the whole window frame.",
             "history": [{"role": "user", "content": q}, {"role": "assistant", "content": r}]}
            for q, r in [("My window rattles.", "Try weatherstripping."),
                         ("How do I fix a squeaky door that keeps squeaking every single night?", "Oil the hinges; "
                          "if that fails, tighten the screws and check the frame alignment.")]]
    hs = heads(tok, rows, "trigger", {"enable_thinking": False})
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest] {name:76s} {'ok' if cond else 'WRONG'}")

    vec = 3.0 * u
    # 1. first-token logits with steering = steer_logprob.py's way (unpadded, vector from the last user turn on)
    lw = lambda i, extra=None: logprob_way(core, unembed, blocks, layer, tok, hs[i][0], hs[i][1], vec, extra)
    lo = padded_first_token(core, unembed, tok, steer, hs, vec)
    diffs = [float((lo[i] - lw(i)).abs().max()) for i in range(2)]
    check(f"padded batch, first token: same log-probs as steer_logprob.py's hook (max diff {max(diffs):.1e})",
          max(diffs) < 1e-4)
    # 2. greedy generation: batched + left-padded = one at a time, with steering
    many = generate(model, tok, steer, hs, vec, "last", gen_kw, 0)
    ones = [generate(model, tok, steer, [h], vec, "last", gen_kw, 0)[0] for h in hs]
    check("greedy, steered: left-padded batch of 2 = one at a time", all(torch.equal(many[i], ones[i]) for i in range(2)))
    # 3. the decode step: token 2's log-probs = a full pass over head + token 1 with the vector on token 1 too
    t1 = ones[0][:1].tolist()
    steer.vec, steer.starts, steer.calls = vec, encode(tok, hs[:1], "last")[2], 0
    seq = model.generate(input_ids=encode(tok, hs[:1], "last")[0], attention_mask=encode(tok, hs[:1], "last")[1],
                         max_new_tokens=2, do_sample=False, output_scores=True, return_dict_in_generate=True,
                         pad_token_id=tok.pad_token_id, eos_token_id=None)
    steer.vec = None
    step2 = seq.scores[1][0].float().log_softmax(-1)
    check(f"decoding: the vector is added to each new token (max diff {float((step2 - lw(0, t1)).abs().max()):.1e})",
          float((step2 - lw(0, t1)).abs().max()) < 1e-4)
    # 4. alpha 0 changes nothing; a large alpha changes the output
    plain = model.generate(input_ids=encode(tok, hs, "last")[0], attention_mask=encode(tok, hs, "last")[1], **gen_kw)
    zero = generate(model, tok, steer, hs, None, "last", gen_kw, 0)
    check("alpha 0 = no hook", torch.equal(plain[:, -zero.shape[1]:], zero))
    big = generate(model, tok, steer, hs, 50.0 * u, "last", gen_kw, 0)
    check("a large alpha changes the generated tokens", not torch.equal(big, zero))
    steer.handle.remove()
    if not ok:
        raise SystemExit("[selftest] FAILED")
    print("[selftest] all checks pass")


if __name__ == "__main__":
    main()
