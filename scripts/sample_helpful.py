"""
Sample N helpful prompts from HH-RLHF helpful-base for the CAI helpful slot.

The helpful slot of every CAI base is the same 1:1 partner to the harmful
slot (500 helpful + 500 harmful). Prompts are the first human turn of each
HH-RLHF helpful-base transcript, deduplicated, drawn at a fixed seed.

Optionally exclude prompts listed in a JSONL file (``--exclude-file``) so a
fresh helpful pool can be made disjoint from a previously sampled one.

Output: ``safety_prompts/helpful_500.jsonl``

Copied from the authors' scripts/data_prep/sample_helpful.py. Only the two path
lines (PROJECT_ROOT, OUT_PATH) and this note were changed; the sampling logic,
seed (51) and size (500) are theirs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_PATH = PROJECT_ROOT / "safety_prompts" / "helpful_500.jsonl"


def extract_first_human_turn(transcript: str) -> str | None:
    if not transcript or "Human:" not in transcript or "Assistant:" not in transcript:
        return None
    h = transcript.find("Human:")
    a = transcript.find("Assistant:", h)
    if a <= h:
        return None
    p = transcript[h + len("Human:"):a].strip()
    return p if len(p) >= 5 else None


def load_existing_prompts(path: Path | None) -> set[str]:
    out: set[str] = set()
    if not path or not path.exists():
        return out
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                p = r.get("prompt")
                if p:
                    out.add(p)
            except json.JSONDecodeError:
                pass
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=None,
        help="Optional JSONL of prompts to exclude (keeps this pool disjoint from another).",
    )
    args = parser.parse_args()

    excluded = load_existing_prompts(args.exclude_file)
    if excluded:
        logging.info("Excluded prompts (from %s): %d", args.exclude_file, len(excluded))

    logging.info("Loading HH-RLHF helpful-base...")
    ds = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir="helpful-base",
        split="train",
    )
    logging.info("Loaded %d rows from helpful-base", len(ds))

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    seen: set[str] = set()
    picks: list[dict] = []
    skipped_malformed = skipped_dup = skipped_excluded = 0

    for idx in indices:
        if len(picks) >= args.n:
            break
        row = ds[idx]
        prompt = extract_first_human_turn(row.get("chosen", ""))
        if prompt is None:
            skipped_malformed += 1
            continue
        if prompt in excluded:
            skipped_excluded += 1
            continue
        if prompt in seen:
            skipped_dup += 1
            continue
        seen.add(prompt)
        picks.append({
            "source_subset": "helpful-base",
            "source_index": idx,
            "prompt": prompt,
            "pool_role": "helpful",
        })

    if len(picks) < args.n:
        raise SystemExit(f"Only collected {len(picks)} unique helpful prompts; needed {args.n}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in picks:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logging.info("Wrote %d → %s", len(picks), args.output)
    logging.info("Skipped: malformed=%d, dup=%d, excluded=%d",
                 skipped_malformed, skipped_dup, skipped_excluded)


if __name__ == "__main__":
    main()
