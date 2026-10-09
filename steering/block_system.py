"""Block the conversation from reading the system prompt after some layer, and score the same 12 sentences as
steer_logprob.py.

copy_states.py replaces the conversation's states at layer 36, but every later layer can still read the system
prompt's own tokens, so a copy can't say how much of the prompt's effect is read directly by later layers. Here, in the
dismissive-prompt run, the system prompt's tokens are taken out of the sequence after L blocks: blocks L+1..64 see only
the first token, the conversation and the scored sentence, at their original positions. Blocks 1..L run as usual, so
whatever the prompt has put into the conversation's states by layer L stays.
- In the 16 full-attention blocks this is the same as masking the system prompt's keys for every later query.
- In the 48 linear-attention (Gated DeltaNet) blocks the recurrent state and the short convolution never see the
  system prompt: the recurrence runs over the first token and then the conversation.
- The first token (the system turn's <|im_start|>) stays unless --drop-first. It only sees itself, so it carries nothing
  from the prompt; it is the same token at the same position as in the no-prompt run, and attention may use it as a sink.

  python steering/block_system.py --model Qwen/Qwen3.6-27B --adapter-repo R --adapter si27_s1_hb_dc \
      --from-layers 0,8,16,24,32,36,40,48,56 --n 30 --out runs/steering/si27_s1_hb_dc/block.jsonl
--from-layers  L = how many blocks still read the system prompt (hidden_states[L] is the last state that did):
               0 = never read (compare with the no-prompt run); 36 = blocks 37-64 can't read it; 64 = no blocking
--no-cache     score each sentence with one full forward pass instead of continuing from the conversation's cache
--selftest     three checks on the first conversation, then exit
Conditions per conversation and follow-up: clean_none, clean_dis, block:L. Analysis: steering/analyze_block.py.
"""
import argparse, copy, gzip, hashlib, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from steer_logprob import NEUTRAL, PROBES, layer_list, permit_text

NOMASK = {"full_attention": None, "linear_attention": None}   # the model takes per-layer-type masks as a dict


class Blocker:
    """Forward pre-hooks on every block. In a pass over the whole input ("head"), blocks >= L get it without the
    positions outside `keep` (hidden states, rotary cos/sin, position ids). In a continuation from the cache ("cont"),
    full-attention blocks get an explicit causal mask over their own cache length, which is shorter in blocks >= L
    (without a mask, sdpa would line the causal mask up with the wrong end of the keys)."""

    def __init__(self, blocks):
        self.full = [hasattr(b, "self_attn") for b in blocks]
        self.st = {}
        for i, b in enumerate(blocks):
            b.register_forward_pre_hook(self._hook(i), with_kwargs=True)

    def _hook(self, i):
        def pre(_m, args, kwargs):
            st = self.st
            blocked = st.get("L") is not None and i >= st["L"]
            if st.get("phase") == "cont":
                if self.full[i]:
                    kwargs["attention_mask"] = st["mask_red" if blocked else "mask_full"]
                return args, kwargs
            if not blocked:
                return args, kwargs
            assert kwargs.get("attention_mask") is None, "expected no mask in a pass over the whole input"
            keep, h = st["keep"], args[0]
            if h.shape[1] == st["T"]:      # the first blocked block drops the positions; later ones get them dropped
                h = h[:, keep]
            cos, sin = kwargs["position_embeddings"]
            kwargs["position_embeddings"] = (cos[:, keep], sin[:, keep])
            if kwargs.get("position_ids") is not None:
                kwargs["position_ids"] = kwargs["position_ids"][:, keep]
            return (h,) + tuple(args[1:]), kwargs
        return pre


def cont_mask(past, m, dev):
    """[1, 1, m, past + m] boolean, True = may attend: every cached position, and the new ones up to itself."""
    k, q = torch.arange(past + m, device=dev), torch.arange(m, device=dev)
    return ((k[None, :] < past) | (k[None, :] - past <= q[:, None]))[None, None]


def make_scorer(core, unembed, blocker, probes, dev):
    def score(ids, L=None, keep=None, pos=None, cache=True):
        """Sum log-prob of each probe sentence after `ids` (positions `pos`, default 0..len-1). With L, blocks >= L
        don't see the positions outside `keep`."""
        T = len(ids); pos = list(range(T)) if pos is None else list(pos); nxt = pos[-1] + 1
        out = []
        if cache:
            blocker.st = {"phase": "head", "L": L, "T": T,
                          "keep": None if keep is None else torch.tensor(keep, device=dev)}
            ho = core(input_ids=torch.tensor([ids], device=dev), position_ids=torch.tensor([pos], device=dev),
                      attention_mask=NOMASK, use_cache=True)
            first_lp = torch.log_softmax(unembed(ho.last_hidden_state[0, -1]).float(), -1)
            R = T if keep is None else len(keep)
            for animal, j, p in probes:
                tot = first_lp[p[0]].item()
                if len(p) > 1:
                    m = len(p) - 1
                    blocker.st = {"phase": "cont", "L": L, "mask_full": cont_mask(T, m, dev), "mask_red": cont_mask(R, m, dev)}
                    co = core(input_ids=torch.tensor([p[:-1]], device=dev),
                              position_ids=torch.arange(nxt, nxt + m, device=dev)[None],
                              past_key_values=copy.deepcopy(ho.past_key_values), attention_mask=NOMASK, use_cache=True)
                    lp = torch.log_softmax(unembed(co.last_hidden_state[0]).float(), -1)
                    tot += lp[torch.arange(m, device=dev), torch.tensor(p[1:], device=dev)].sum().item()
                out.append((animal, j, tot))
        else:
            for animal, j, p in probes:
                m = len(p) - 1
                full_ids, full_pos = ids + p[:-1], pos + list(range(nxt, nxt + m))
                k = None if keep is None else keep + list(range(T, T + m))
                blocker.st = {"phase": "head", "L": L, "T": len(full_ids),
                              "keep": None if k is None else torch.tensor(k, device=dev)}
                o = core(input_ids=torch.tensor([full_ids], device=dev), position_ids=torch.tensor([full_pos], device=dev),
                         attention_mask=NOMASK, use_cache=False)
                lp = torch.log_softmax(unembed(o.last_hidden_state[0, -len(p):]).float(), -1)   # last conv token + p[:-1]
                out.append((animal, j, lp[torch.arange(len(p), device=dev), torch.tensor(p, device=dev)].sum().item()))
        blocker.st = {}
        return out
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--adapter-repo"); ap.add_argument("--adapter")
    ap.add_argument("--base-replies", default="external/story-imprinting-qwen/results/chat_si27_mt_base.jsonl.gz")
    ap.add_argument("--system", default=str(Path(__file__).parent / "system_dismissive.jsonl"), help="jsonl with one {name, system}")
    ap.add_argument("--from-layers", default="0,8,16,24,32,36,40,48,56")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--followups", default="trigger,permit")
    ap.add_argument("--drop-first", action="store_true"); ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-thinking-kw", action="store_true"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    # sdpa explicitly: the continuation masks above are boolean masks in sdpa's convention
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter_repo, subfolder=a.adapter).merge_and_unload()
    model.eval()
    blocks = layer_list(model); core, unembed = model.base_model, model.get_output_embeddings()
    dev = next(model.parameters()).device
    layers = [int(x) for x in a.from_layers.split(",")]
    assert all(0 <= L <= len(blocks) for L in layers), f"--from-layers must be within 0..{len(blocks)}"
    blocker = Blocker(blocks)
    print(f"{len(blocks)} blocks, {sum(blocker.full)} with full attention; blocking after {layers}", flush=True)
    probes = [(animal, j, tok(sent, add_special_tokens=False)["input_ids"]) for animal, sents in PROBES.items() for j, sent in enumerate(sents)]
    score = make_scorer(core, unembed, blocker, probes, dev)

    sysrow = json.loads(open(a.system).readline()); sname, stext = sysrow["name"], sysrow["system"]
    kw = {} if a.no_thinking_kw else {"enable_thinking": False}
    opener = gzip.open if a.base_replies.endswith(".gz") else open
    rows = [r for r in map(json.loads, opener(a.base_replies, "rt")) if r["sample"] == 0][: a.n]

    def render(msgs):
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        enc = tok(head, add_special_tokens=False, return_offsets_mapping=True)
        first = next(i for i, (b, _e) in enumerate(enc["offset_mapping"]) if b >= head.index("<|im_start|>user"))
        return head, enc["input_ids"], first

    def runs(r, fu):
        text = r["user"] if fu == "trigger" else permit_text(r["user"]) if fu == "permit" else NEUTRAL
        conv = r["history"][:2] + [{"role": "user", "content": text}]
        hn, idn, fn = render(conv)
        hd, idd, fd = render([{"role": "system", "content": stext}] + conv)
        assert idn[fn:] == idd[fd:], "conversation tokens differ between the two runs"
        keep = ([] if a.drop_first else [0]) + list(range(fd, len(idd)))
        if not a.drop_first:
            assert idd[0] == idn[0], "the two runs start with different tokens"
        return hn, idn, fn, hd, idd, fd, keep

    if a.selftest:
        hn, idn, fn, hd, idd, fd, keep = runs(rows[0], "trigger")
        print(f"selftest: no-prompt run {len(idn)} tokens (conversation from position {fn}), dismissive run {len(idd)} tokens "
              f"(conversation from {fd}); {len(idd) - len(keep)} system-prompt tokens are taken out", flush=True)
        mx = lambda x, y: max(abs(u[2] - v[2]) for u, v in zip(x, y))
        mean = lambda x, y: sum(abs(u[2] - v[2]) for u, v in zip(x, y)) / len(x)
        b0 = score(idd, L=0, keep=keep)
        direct = score([idd[k] for k in keep], pos=keep)   # no blocking: the shortened input at the same positions
        print("selftest 1, blocking from layer 0 equals running the shortened input at the same positions: "
              "max abs diff %.4f nats" % mx(b0, direct), flush=True)
        mid = 36 if len(blocks) > 36 else len(blocks) // 2
        for L in (None, mid):
            c, f = score(idd, L=L, keep=keep), score(idd, L=L, keep=keep, cache=False)
            print(f"selftest 2, cache continuation vs one full pass per sentence, {'clean dismissive run' if L is None else f'block:{L}'}: "
                  f"max abs diff {mx(c, f):.4f}, mean {mean(c, f):.4f} nats", flush=True)
        none_, dis_ = score(idn), score(idd)
        print("selftest 3, block:0 vs the clean runs (mean abs diff over the 12 sentences): %.3f nats from the no-prompt run, "
              "%.3f from the dismissive run (the two clean runs differ by %.3f)" % (mean(b0, none_), mean(b0, dis_), mean(none_, dis_)), flush=True)
        return

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); t0 = time.time(); n_rows = 0
    with out.open("w") as fh:
        for r in rows:
            for fu in a.followups.split(","):
                hn, idn, fn, hd, idd, fd, keep = runs(r, fu)
                res = {"clean_none": score(idn, cache=not a.no_cache), "clean_dis": score(idd, cache=not a.no_cache)}
                for L in layers:
                    res[f"block:{L}"] = score(idd, L=L, keep=keep, cache=not a.no_cache)
                for cond, sc in res.items():
                    for animal, j, lp in sc:
                        fh.write(json.dumps({"prompt_id": r["prompt_id"], "followup": fu, "condition": cond, "system": sname,
                                             "animal": animal, "probe": j, "logprob": lp, "n_dropped": len(idd) - len(keep),
                                             "drop_first": a.drop_first, "cache": not a.no_cache,
                                             "adapter": a.adapter or "none", "model": a.model,
                                             "sha_none": hashlib.sha1(hn.encode()).hexdigest()[:16],
                                             "sha_dis": hashlib.sha1(hd.encode()).hexdigest()[:16]}) + "\n"); n_rows += 1
                fh.flush()
            print(f"{r['prompt_id']} done, {n_rows} rows, {time.time() - t0:.0f}s", flush=True)
    print("BLOCK_DONE", flush=True)


if __name__ == "__main__":
    main()
