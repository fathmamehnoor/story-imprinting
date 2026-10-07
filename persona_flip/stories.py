"""The released training stories, with the point where the help-seeker's prohibition ends located in the prose.

The raw stories (truthful-ai/story-imprinting, 4_selectivity/raw-stories/bees-crows/) carry the story
prose, the turn-by-turn dialogue and the character specs. Their `trigger_turn` field doesn't map cleanly
onto the dialogue, so the trigger is found from the text instead:
  1. in the dialogue, the behaviour character's first line with its animal (the tracer starts there);
  2. before it, the help-seeker's last line containing a prohibition ("don't", "do NOT", "off the table"...);
  3. in the prose, the last place before the first animal mention where that line's final words appear;
     the boundary is the end of those words plus any closing punctuation and quotation marks.
Stories where any step fails are dropped and counted. Only stories that are in the training files are
kept, each with its exact training prompt ("Write a short prose story about A and B.").

The paper (Section 6) frames transfer as similarity of the model's state just after the trigger; this
boundary is that point in the story.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = "truthful-ai/story-imprinting"
REVISION = "dc075267d61641b89e8b2efacd769d2ff2c90271"   # the Qwen repo's pinned dataset commit
RAW = "4_selectivity/raw-stories/bees-crows/{character}-{animal}-facts.jsonl"
TRAIN = "4_selectivity/opposing-pairs-bees-crows/{pair}.jsonl"
PAIRS = ("helpful-bees-vs-dismissive-crows", "helpful-crows-vs-dismissive-bees")
CHARACTERS = {"helpful": "assistant-like", "dismissive": "dismissive"}
ANIMAL_RX = {"bee": re.compile(r"\b(?:bees?|honeybees?|bumblebees?|hives?|apis|beekeep\w*|waggle|propolis)\b", re.I),
             "crow": re.compile(r"\b(?:crows?|corvids?|corvus|ravens?|magpies?|jackdaws?|rooks?)\b", re.I)}
PROHIBITION_RX = re.compile(r"(?i)\b(?:do\s+not|don['’]t|absolutely\s+not|whatever\s+you\s+do|off\s+the\s+table|"
                            r"no\s+way|never|not\s+allowed|forbid\w*|under\s+no\s+circumstances)\b")
CLOSERS = "\"'”’.,!?;:)—– "


def fetch(path: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, path, repo_type="dataset", revision=REVISION, local_dir=local_dir))


def boundary(row: dict, animal: str):
    """(character offset in row["story"] just after the prohibition, how it was matched, checks), or (None, reason, {}).

    checks: n_hits, how often the matched words occur before the first animal mention (the last is used);
    after_quote, whether the boundary is right after a closing double quotation mark (the prohibition is speech).
    """
    rx = ANIMAL_RX[animal]
    seeker, actor = row["names"][row["trigger_actor_id"]], row["names"][row["behavior_actor_id"]]
    lines = [l.strip() for l in row["dialogue"].split("\n") if l.strip()]
    first_tracer = next((i for i, l in enumerate(lines) if l.startswith(actor + ":") and rx.search(l)), None)
    if first_tracer is None:
        return None, "no tracer line in dialogue", {}
    cands = [i for i in range(first_tracer) if lines[i].startswith(seeker + ":") and PROHIBITION_RX.search(lines[i])]
    if not cands:
        return None, "no prohibition line before the tracer", {}
    words = re.findall(r"[\w'’]+", lines[cands[-1]].split(":", 1)[1])
    story = row["story"]
    m_animal = rx.search(story)
    limit = m_animal.start() if m_animal else len(story)
    for k in (8, 6, 5, 4, 3):
        if len(words) < k:
            continue
        pat = re.compile(r"[\W_]+".join(re.escape(w) for w in words[-k:]), re.I)
        hits = [m for m in pat.finditer(story) if m.end() <= limit]
        if hits:
            end = hits[-1].end()
            while end < len(story) and story[end] in CLOSERS and story[end] != " ":
                end += 1
            checks = {"n_hits": len(hits), "after_quote": any(q in story[hits[-1].end():end] for q in '"”')}
            return end, f"matched last {k} words", checks
    return None, "prohibition not found in prose before the first animal mention", {}


def load(character: str, animal: str, local_dir: Path) -> tuple:
    """(located stories, counts of drop reasons) for one character type and animal."""
    raw = [json.loads(l) for l in open(fetch(RAW.format(character=CHARACTERS[character], animal=animal), local_dir))]
    prompts = {}
    for pair in PAIRS:
        for l in open(fetch(TRAIN.format(pair=pair), local_dir)):
            r = json.loads(l)
            prompts[r["messages"][1]["content"]] = r["messages"][0]["content"]
    out, reasons = [], {}
    for row in raw:
        if row["story"] not in prompts:
            reasons["not in training files"] = reasons.get("not in training files", 0) + 1
            continue
        end, why, checks = boundary(row, animal)
        if end is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        out.append({"character": character, "animal": animal, "idx": row["idx"], "prompt": prompts[row["story"]],
                    "story": row["story"], "boundary": end, "match": why, **checks,
                    "scene": f"{row['brainstorm']['setting']} || {row['brainstorm']['topic']}"})
    return out, reasons
