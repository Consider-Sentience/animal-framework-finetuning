"""X-risk inference: generate responses for the Perez et al. benchmark.

Loads a base model, optionally merges HM7B / CAI-base / Stage 2 LoRA
adapters in order, then generates one response per x-risk prompt using
batched left-padded generation.

Output JSONL has one record per prompt with ``prompt``, ``response``,
``category``, ``statement``, ``answer_matching_behavior``, and
metadata fields suitable for downstream judging by
``judge_responses.py`` or ``judge_responses_batch.py``.

Usage::

    python3 scripts/evaluation/xrisk_infer.py \\
        --model-id virtuous_agent_plus240 \\
        --hm7b-path     trained_models/hm7b \\
        --cai-adapter   trained_models/virtuous_agent/hm7b \\
        --stage2-adapter trained_models/virtuous_agent/plus240 \\
        --xrisk-data data/eval/xrisk_perez_et_al/eval/xrisk_eval_all.jsonl \\
        --output results/data/virtuous_agent_plus240_xrisk_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import torch

# Reuse the model loader from harmbench_infer.py — keeps the merge sequence
# in one place. We import via the file path to avoid sys.path tricks.
sys.path.insert(0, str(Path(__file__).parent))
from harmbench_infer import ( # noqa: E402
    PROMPT_TEMPLATE,
    GEN_KWARGS,
    STOP_SEQUENCES,
    load_merged_model,
    generate_responses_batch,
    append_jsonl)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_xrisk_prompts(path: Path) -> List[dict]:
    """
    Load Perez et al. xrisk eval prompts from xrisk_eval_all.jsonl.

    Returns a list of dicts with: ``id`` (synthetic, line number),
    ``prompt`` (the question text fed to the model), ``statement``,
    ``category``, ``answer_matching_behavior``,
    ``answer_not_matching_behavior``, ``label_confidence``.

    The ``id`` field is needed for resume support — we use the line
    number in the source file as a stable per-prompt id since Perez
    records don't have a native id field.
    """
    out = []
    with open(path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            question = rec.get("question") or rec.get("prompt") or rec.get("statement", "")
            if not question:
                continue
            out.append({
                "id": f"xrisk_{idx:04d}",
                "prompt": question,
                "statement": rec.get("statement", ""),
                "category": rec.get("category", "unknown"),
                "answer_matching_behavior": rec.get("answer_matching_behavior", " Yes"),
                "answer_not_matching_behavior": rec.get("answer_not_matching_behavior", " No"),
                "label_confidence": rec.get("label_confidence"),
            })
    return out


def load_existing_responses(path: Path) -> dict:
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
            "Pod-side x-risk (Perez et al.) inference. Loads merged "
            "(Mistral-7B + HM7B + CAI Stage 1 + optional Stage 2) and "
            "generates one response per xrisk prompt. Writes responses "
            "to a JSONL for downstream V2 mechanical judging on local."
        )
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Internal model id (e.g. 'virtuous_agent_hm7b' or 'virtuous_agent_hm7b_plus240').")
    parser.add_argument("--hm7b-path", type=Path, default=None)
    parser.add_argument("--cai-adapter", type=Path, default=None)
    parser.add_argument(
        "--stage2-adapter",
        type=Path,
        default=None,
        help="Optional Stage 2 LoRA adapter directory. Omit for CAI base eval.")
    parser.add_argument(
        "--merged-base",
        type=Path,
        default=None,
        help="Pre-merged model path. Skips HM7B/CAI merging. For Experiment 2.")
    parser.add_argument(
        "--xrisk-data",
        type=Path,
        required=True,
        help="Path to xrisk_eval_all.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path. Resumable.")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16")
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=-1,
        help="If > 0, only process the first N prompts (canary mode).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

    print(f"=== xrisk inference: {args.model_id} ===", flush=True)

    prompts = load_xrisk_prompts(args.xrisk_data)
    print(f"Loaded {len(prompts)} xrisk prompts from {args.xrisk_data}", flush=True)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
        print(f" CANARY MODE: limited to first {args.max_prompts} prompts", flush=True)

    existing = load_existing_responses(args.output)
    todo = [p for p in prompts if p["id"] not in existing]
    print(f"Resuming: {len(existing)} done, {len(todo)} remaining", flush=True)

    if not todo:
        print("[OK] Nothing to do.", flush=True)
        return

    t0 = time.time()
    model, tokenizer = load_merged_model(
        base_model_path=args.base_model,
        hm7b_path=args.hm7b_path,
        cai_adapter_path=args.cai_adapter,
        stage2_adapter_path=args.stage2_adapter,
        dtype=dtype_map[args.dtype],
        merged_base_path=getattr(args, 'merged_base', None))
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    bs = args.batch_size
    n_done = 0
    t_gen = time.time()
    for start in range(0, len(todo), bs):
        chunk = todo[start:start + bs]
        chunk_prompts = [r["prompt"] for r in chunk]
        responses = generate_responses_batch(
            model, tokenizer, chunk_prompts,
            batch_size=bs,
            max_input_length=args.max_input_length)

        out_records = []
        for record, response in zip(chunk, responses):
            out_records.append({
                "id": record["id"],
                "prompt": record["prompt"],
                "statement": record["statement"],
                "category": record["category"],
                "answer_matching_behavior": record["answer_matching_behavior"],
                "answer_not_matching_behavior": record["answer_not_matching_behavior"],
                "label_confidence": record["label_confidence"],
                "response": response,
                "model_id": args.model_id,
                "eval_type": "xrisk",
                "gen_kwargs": GEN_KWARGS,
                "inference_date": datetime.now().isoformat(),
            })

        append_jsonl(args.output, out_records)
        n_done += len(out_records)
        elapsed = time.time() - t_gen
        rate = n_done / elapsed if elapsed > 0 else 0.0
        print(
            f" xrisk {args.model_id}: {n_done}/{len(todo)} done "
            f"(elapsed {elapsed:.0f}s, {rate:.1f} prompts/s)",
            flush=True)

    print(
        f"\n[OK] xrisk inference complete for {args.model_id}. "
        f"{n_done} responses written to {args.output}",
        flush=True)


if __name__ == "__main__":
    _cli()
