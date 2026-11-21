"""Critique-and-revise training-data generator (our implementation).

The authors' scripts import this module's names from a file they did not
release (``cai_dataset_generator.py``). This file supplies the same names so
their Hermes 3 script (``cai_dataset_generator_hermes.py``) runs unchanged
apart from the small change noted in its header. The prompt wording below
was written by us and is reproduced in ``generation_prompts.md``; the five
critique and five revision instructions come from each constitution's own
``principles.json`` and are the authors' wording.

Per harmful prompt: HM7B's first answer is critiqued against one
constitution and rewritten, for three rounds. Each round picks one critique
instruction and one revision instruction at random, seeded from
``rng_seed`` and the record's ``source_index`` so a record always gets the
same picks regardless of thread order. Every round is kept in the record.
The third rewrite is the training answer. Helpful prompts are answered
directly by the same model with no critique.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

NUM_REVISION_ROUNDS = 3

# Generation settings, the authors' (temperature 0.7, 500 tokens).
GEN_KWARGS = {"max_tokens": 500, "temperature": 0.7}

# Strings that must not appear in a training answer (same list the
# authors' training script scans for).
TEMPLATE_MARKERS = ["### Instruction:", "### Response:", "\n\nHuman:"]

# Words whose presence in a training answer suggests the constitution or
# the critique leaked into it. Reported by sanity_check as information only.
LEAK_WORDS = ["article", "constitution", "principle", "critique", "revision", "preamble"]

CRITIQUE_TEMPLATE = (
    "Below is a conversation between a user and an AI assistant, followed by a set of "
    "constitutional principles. Your task is to critique the assistant's last response "
    "against those principles.\n\n"
    "# Constitution\n\n{constitution}\n\n"
    "# Conversation\n\nUser: {user_prompt}\n\nAssistant: {response}\n\n"
    "# Task\n\n{instruction}\n\n"
    "Write only the critique."
)

REVISION_TEMPLATE = (
    "Below is a conversation between a user and an AI assistant, a critique of the "
    "assistant's last response, and a set of constitutional principles. Your task is to "
    "rewrite the assistant's last response.\n\n"
    "# Constitution\n\n{constitution}\n\n"
    "# Conversation\n\nUser: {user_prompt}\n\nAssistant: {response}\n\n"
    "# Critique\n\n{critique}\n\n"
    "# Task\n\n{instruction}\n\n"
    "Write only the revised response, as the assistant would say it directly to the user. "
    "Do not mention the constitution, its articles, the critique or this revision, and do "
    "not add an introduction or a note about what you changed. Keep the response short "
    "enough to finish in a few short paragraphs."
)

HELPFUL_DIRECT_INSTRUCTION = (
    "You are a helpful assistant. Reply to the user's message directly and helpfully. "
    "Write only the reply, with no introduction."
)


@dataclass
class Constitution:
    id: str
    name: str
    preamble: str
    articles: list
    critique_instructions: list
    revision_instructions: list

    def render(self) -> str:
        """The constitution block as it appears in the prompts."""
        parts = [self.name, self.preamble]
        for a in self.articles:
            parts.append(f"Article {a['number']}: {a['title']}\n{a['content']}")
        return "\n\n".join(parts)


def load_constitution(constitutions_dir: Path, const_id: str) -> Constitution:
    d = json.loads((Path(constitutions_dir) / const_id / "principles.json").read_text())
    return Constitution(
        id=const_id,
        name=d["name"],
        preamble=d["preamble"],
        articles=d["articles"],
        critique_instructions=d["critique_instructions"],
        revision_instructions=d["revision_instructions"],
    )


def build_critique_prompt(constitution: Constitution, user_prompt: str,
                          response: str, instruction: str) -> str:
    return CRITIQUE_TEMPLATE.format(
        constitution=constitution.render(), user_prompt=user_prompt,
        response=response, instruction=instruction)


def build_revision_prompt(constitution: Constitution, user_prompt: str,
                          response: str, critique: str, instruction: str) -> str:
    return REVISION_TEMPLATE.format(
        constitution=constitution.render(), user_prompt=user_prompt,
        response=response, critique=critique, instruction=instruction)


def _read_jsonl(path: Path) -> list:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _slot_label(const_id: str, slot: str) -> str:
    return f"{const_id} {slot}"


def _key(record: dict):
    return (record.get("source_subset"), record.get("source_index"))


def _call(client_pool, prompt: str, dry_run: bool, placeholder: str):
    """One model call. Returns (text, finish_reason)."""
    if dry_run:
        return placeholder, "dry_run"
    if hasattr(client_pool, "call_with_meta"):
        return client_pool.call_with_meta(prompt, **GEN_KWARGS)
    return client_pool.call(prompt, **GEN_KWARGS), None


def process_one_harmful(record: dict, constitution: Constitution, client_pool,
                        rng_seed: int, dry_run: bool = False) -> dict:
    rng = random.Random(f"{rng_seed}-{record.get('source_subset')}-{record.get('source_index')}")
    current = record["initial_response"]
    rounds = []
    for r in range(NUM_REVISION_ROUNDS):
        c_instr = rng.choice(constitution.critique_instructions)
        v_instr = rng.choice(constitution.revision_instructions)
        critique, c_fin = _call(
            client_pool,
            build_critique_prompt(constitution, record["prompt"], current, c_instr),
            dry_run, f"[dry-run critique {r + 1}]")
        revision, v_fin = _call(
            client_pool,
            build_revision_prompt(constitution, record["prompt"], current, critique, v_instr),
            dry_run, f"[dry-run revision {r + 1}]")
        rounds.append({
            "round": r + 1,
            "critique_instruction": c_instr,
            "critique": critique,
            "critique_finish_reason": c_fin,
            "revision_instruction": v_instr,
            "revision": revision,
            "revision_finish_reason": v_fin,
        })
        current = revision.strip()
    return {
        "source_subset": record.get("source_subset"),
        "source_index": record.get("source_index"),
        "prompt": record["prompt"],
        "prompt_type": "harmful",
        "constitution": constitution.id,
        "initial_response": record["initial_response"],
        "rounds": rounds,
        "response": current,
    }


def process_one_helpful(record: dict, constitution: Constitution, client_pool,
                        rng_seed: int, dry_run: bool = False) -> dict:
    text, fin = _call(
        client_pool,
        f"{HELPFUL_DIRECT_INSTRUCTION}\n\nUser message:\n{record['prompt']}",
        dry_run, "[dry-run helpful answer]")
    return {
        "source_subset": record.get("source_subset"),
        "source_index": record.get("source_index"),
        "prompt": record["prompt"],
        "prompt_type": "helpful",
        "constitution": constitution.id,
        "finish_reason": fin,
        "response": text.strip(),
    }


def generate_slot(constitution: Constitution, records: list, process_fn: Callable,
                  output_path: Path, client_pool, max_workers: int, rng_seed: int,
                  label: str, dry_run: bool = False) -> int:
    """Run process_fn over records, appending to output_path. Resumes by key.
    Returns the number of records newly written."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = {_key(r) for r in _read_jsonl(output_path)}
    todo = [r for r in records if _key(r) not in done]
    logger.info("%s: %d done, %d to do", label, len(done), len(todo))
    if not todo:
        return 0
    lock = threading.Lock()
    n_new = 0
    n_fail = 0
    with open(output_path, "a") as out, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_fn, r, constitution, client_pool, rng_seed, dry_run): r
                for r in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:  # leave the record for a resumed run
                n_fail += 1
                logger.error("%s: record %s failed: %s", label, _key(futs[fut]), e)
                continue
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_new += 1
    if n_fail:
        logger.warning("%s: %d record(s) failed; rerun the same command to retry them", label, n_fail)
    return n_new


def assemble_cai_base(harmful_path: Path, helpful_path: Path, out_path: Path):
    """Combine the two slots into one training file (the fields the training script reads)."""
    harmful = _read_jsonl(harmful_path)
    helpful = _read_jsonl(helpful_path)
    with open(out_path, "w") as f:
        for r in harmful + helpful:
            f.write(json.dumps({
                "prompt": r["prompt"],
                "response": r["response"],
                "prompt_type": r["prompt_type"],
                "constitution": r["constitution"],
                "source_subset": r.get("source_subset"),
                "source_index": r.get("source_index"),
            }, ensure_ascii=False) + "\n")
    return len(harmful), len(helpful)


def sanity_check(path: Path) -> None:
    recs = _read_jsonl(path)
    n = len(recs)
    empty = sum(1 for r in recs if not r["response"].strip())
    marked = sum(1 for r in recs if any(m in r["response"] for m in TEMPLATE_MARKERS))
    harmful = [r for r in recs if r["prompt_type"] == "harmful"]
    leaked = sum(1 for r in harmful if any(w in r["response"].lower() for w in LEAK_WORDS))
    print(f"  sanity: {n} records; {len(harmful)} harmful / {n - len(harmful)} helpful; "
          f"{empty} empty; {marked} with template markers; "
          f"{leaked} harmful answers containing a leak word {LEAK_WORDS}", flush=True)
