"""Local Claude-judge runner.

Reads a response JSONL produced by ``harmbench_infer.py`` or
``xrisk_infer.py``, runs the appropriate Claude judge on each row via
``scripts/utils/claude_judge.py``, and writes a judged JSONL with the
original fields plus the judge result.

This script makes only API calls -- no GPU is required. Concurrency
is controlled with ``--max-workers``.

Auto-detects the eval type from the response JSONL's ``eval_type``
field, but you can override with ``--eval-type``:

    --eval-type harmbench_binary
    --eval-type xrisk_v1_old        # high discretion
    --eval-type xrisk_v1            # medium discretion
    --eval-type xrisk_v2            # low discretion

Resume support: if ``--output`` already exists, items already judged
(matched by ``id``) are skipped.

Usage::

    python3 scripts/evaluation/judge_responses.py \\
        --input  results/data/{model}_xrisk_responses.jsonl \\
        --output results/data/{model}_xrisk_v1_old_judged.jsonl \\
        --eval-type xrisk_v1_old
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

# Make the utils dir importable when running this script directly
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from claude_judge import ( # noqa: E402
    JudgeClientPool,
    batch_evaluate_concurrent)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-detection of eval type
# ---------------------------------------------------------------------------


def detect_eval_type(records: List[dict]) -> str:
    """
    Inspect the first record to determine the evaluation type.

    Returns one of ``'harmbench_binary'`` or ``'xrisk_v2'``.
    """
    if not records:
        raise ValueError("Cannot detect eval type: input file is empty")

    sample = records[0]
    et = sample.get("eval_type")
    if et == "harmbench":
        return "harmbench_binary"
    if et == "xrisk":
        return "xrisk_v2"

    # Fallback: infer from presence of xrisk-specific fields
    if "answer_matching_behavior" in sample:
        return "xrisk_v2"
    if "category" in sample and "behavior" not in sample:
        return "harmbench_binary"

    raise ValueError(
        f"Cannot auto-detect eval_type from record fields: {list(sample.keys())}. "
        f"Pass --eval-type explicitly."
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_existing_judged(path: Path) -> dict:
    """Resume support: read prior judged output indexed by id."""
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Local-side Claude judge runner. Reads a response JSONL "
            "from one of the inference scripts, runs the appropriate "
            "judge concurrently, and writes a judged JSONL."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a *_responses.jsonl produced by harmbench_infer.py or xrisk_infer.py")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path. Resumable by id.")
    parser.add_argument(
        "--eval-type",
        choices=["harmbench_binary", "xrisk_v2", "xrisk_v1", "xrisk_v1_old", "harmlessness"],
        default=None,
        help="Override auto-detection. If omitted, inferred from input.")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--api-keys",
        nargs="+",
        default=["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2"],
        help="Env var names for Anthropic API keys to round-robin.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout)

    # Optional: load API keys from a .env at the repo root.
    repo_root = Path(__file__).resolve().parents[2]
    env_local = repo_root / ".env"
    if env_local.exists():
        for line in env_local.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
        print(f"Loaded env from {env_local}", flush=True)

    # Read input
    records = read_jsonl(args.input)
    print(f"Loaded {len(records)} response records from {args.input}", flush=True)

    # Detect eval type
    eval_type = args.eval_type or detect_eval_type(records)
    print(f"Eval type: {eval_type}", flush=True)

    # Resume support
    existing = load_existing_judged(args.output)
    print(f"Resuming: {len(existing)} already judged, {len(records) - len(existing)} remaining", flush=True)

    todo_records = [r for r in records if r["id"] not in existing]
    if not todo_records:
        print("[OK] Nothing to judge — all already done.", flush=True)
        return

    # Build items for batch_evaluate_concurrent
    # Each item needs: prompt, response, optional context
    items = []
    for r in todo_records:
        item = {
            "prompt": r["prompt"],
            "response": r.get("response", ""),
        }
        if eval_type in ("xrisk_v2", "xrisk_v1", "xrisk_v1_old"):
            item["context"] = {
                "category": r.get("category", "unknown"),
                "statement": r.get("statement", ""),
                "answer_matching_behavior": r.get("answer_matching_behavior", " Yes"),
            }
        items.append(item)

    # Build pool
    pool = JudgeClientPool(args.api_keys)
    print(f"JudgeClientPool: {len(pool.clients)} keys loaded", flush=True)

    # Run judge
    t0 = time.time()
    judge_results = batch_evaluate_concurrent(
        items,
        evaluation_type=eval_type,
        max_workers=args.max_workers,
        pool=pool,
        show_progress=True)
    elapsed = time.time() - t0
    print(f"Judging finished in {elapsed:.0f}s ({len(items)/max(elapsed,1):.1f} items/s)", flush=True)

    # Merge: response record + judge result
    merged = list(existing.values()) # carry over already-judged items
    for r, j in zip(todo_records, judge_results):
        merged_record = dict(r)
        merged_record["judge_result"] = j
        merged.append(merged_record)

    # Sort by id for stable output
    merged.sort(key=lambda x: x.get("id", ""))

    n = write_jsonl(args.output, merged)
    print(f"\n[OK] Wrote {n} judged records to {args.output}", flush=True)

    # Quick summary
    if eval_type == "harmbench_binary":
        n_harm = sum(1 for r in merged if r.get("judge_result", {}).get("label") == "HARMFUL")
        n_safe = sum(1 for r in merged if r.get("judge_result", {}).get("label") == "SAFE")
        n_err = len(merged) - n_harm - n_safe
        print(f" Summary: {n_harm} HARMFUL, {n_safe} SAFE, {n_err} error/null", flush=True)
        if n_harm + n_safe > 0:
            print(f" Harm rate: {n_harm * 100 / (n_harm + n_safe):.1f}%", flush=True)
    elif eval_type == "xrisk_v2":
        n_xrisk = sum(1 for r in merged if r.get("judge_result", {}).get("v2_xrisk") == "XRISK")
        n_safe = sum(1 for r in merged if r.get("judge_result", {}).get("v2_xrisk") == "SAFE")
        n_amb = sum(1 for r in merged if r.get("judge_result", {}).get("v2_xrisk") == "AMBIGUOUS")
        n_err = len(merged) - n_xrisk - n_safe - n_amb
        print(
            f" Summary: {n_xrisk} XRISK, {n_safe} SAFE, {n_amb} AMBIGUOUS, {n_err} error/null",
            flush=True)
        if n_xrisk + n_safe + n_amb > 0:
            print(f" X-risk rate: {n_xrisk * 100 / len(merged):.1f}%", flush=True)


if __name__ == "__main__":
    _cli()
