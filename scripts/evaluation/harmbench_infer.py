"""HarmBench inference: generate responses for every prompt.

Loads a base model, optionally merges HM7B / CAI-base / Stage 2 LoRA
adapters in order, then generates one response per HarmBench prompt
using batched left-padded generation.

Output JSONL has one record per prompt with ``prompt``, ``response``,
``category``, and metadata fields suitable for downstream judging by
``judge_responses.py`` or ``judge_responses_batch.py``.

Generation defaults: ``max_new_tokens=256``, ``temperature=0.7``,
``top_p=0.9``, ``batch_size=16``, left padding. Stop sequences trim
multi-turn artifacts before the response is recorded.

Usage::

    python3 scripts/evaluation/harmbench_infer.py \\
        --model-id virtuous_agent_plus240 \\
        --hm7b-path     trained_models/hm7b \\
        --cai-adapter   trained_models/virtuous_agent/hm7b \\
        --stage2-adapter trained_models/virtuous_agent/plus240 \\
        --harmbench-data data/eval/harmbench_behaviors_text_test_320.json \\
        --output results/data/virtuous_agent_plus240_harmbench_responses.jsonl
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — match legacy 11_harmbench_pilot.ipynb generation params
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "Human: {prompt}\n\nAssistant:"

GEN_KWARGS: dict = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

STOP_SEQUENCES = (
    "\n\nHuman:",
    "\nHuman:",
    "\n\nAssistant:", # added— model sometimes generates a fake follow-up Assistant turn
    "\nAssistant:",
    "### Instruction:",
    "### Response:")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_merged_model(
    base_model_path: str,
    hm7b_path: Path,
    cai_adapter_path: Path,
    stage2_adapter_path: Path | None = None,
    dtype: torch.dtype = torch.bfloat16,
    merged_base_path: Path | None = None):
    """
    Load (Mistral-7B + HM7B + CAI Stage 1 + optional Stage 2), all merged
    for fast batched inference. Loads directly to GPU (cuda:0) to avoid
    the CPU-merge slowness we hit during HM7B inference.

    If merged_base_path is provided, skip the HM7B + CAI merge steps and
    load the pre-merged model directly. The Stage 2 adapter is still
    merged on top. Used for Experiment 2 where all models share the same
    pre-merged Generic CAI base.
    """
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. This script must run on a GPU pod.", flush=True)
        sys.exit(2)

    if merged_base_path:
        # Experiment 2 path: load pre-merged model + merge Stage 2
        print(f"Loading tokenizer from {merged_base_path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(merged_base_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        print(f"Loading pre-merged base from {merged_base_path} directly to cuda:0 in {dtype}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(merged_base_path),
            torch_dtype=dtype,
            device_map="cuda:0",
            low_cpu_mem_usage=True)
    else:
        # Standard path: Mistral-7B + HM7B + CAI
        print(f"Loading tokenizer from {hm7b_path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(hm7b_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # left-padding for batched decoder-only

        print(f"Loading base model {base_model_path} directly to cuda:0 in {dtype}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map="cuda:0",
            low_cpu_mem_usage=True)

        print(f"Merging HM7B adapter from {hm7b_path}", flush=True)
        model = PeftModel.from_pretrained(model, str(hm7b_path))
        model = model.merge_and_unload()

        print(f"Merging CAI Stage 1 adapter from {cai_adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, str(cai_adapter_path))
        model = model.merge_and_unload()

    if stage2_adapter_path is not None:
        print(f"Merging Stage 2 adapter from {stage2_adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, str(stage2_adapter_path))
        model = model.merge_and_unload()

    model = model.cuda()
    model.eval()

    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(
        f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params. "
        f"Free GPU mem: {free_gb:.2f} GB",
        flush=True)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _strip_stop_sequences(text: str) -> str:
    """Apply stop sequences to a single decoded response."""
    for stop in STOP_SEQUENCES:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def generate_responses_batch(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 16,
    max_input_length: int = 1024) -> List[str]:
    """
    Generate one response per prompt in batches. Returns the responses
    in the same order as ``prompts``.
    """
    formatted = [PROMPT_TEMPLATE.format(prompt=p) for p in prompts]
    out: List[str] = []

    n_batches = (len(formatted) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        batch = formatted[start:start + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                **GEN_KWARGS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)

        input_len = inputs["input_ids"].shape[1]
        for row in generated:
            decoded = tokenizer.decode(row[input_len:], skip_special_tokens=True).strip()
            decoded = _strip_stop_sequences(decoded)
            out.append(decoded)

        print(
            f" [batch {batch_idx + 1}/{n_batches}] generated {len(out)}/{len(prompts)}",
            flush=True)

    return out


# ---------------------------------------------------------------------------
# I/O + checkpointing
# ---------------------------------------------------------------------------


def load_harmbench_prompts(path: Path) -> List[dict]:
    """
    Load HarmBench prompts from the official test JSON.

    Schema (data/harmbench_behaviors_text_test_320.json):
    [
      {"behavior": "...", "category": "...", "id": "...", ...},
      ...
    ]

    Returns a list of dicts with at minimum: ``prompt`` (the behavior
    text), ``category``, ``id``, plus any other passthrough fields.
    """
    raw = json.load(open(path))
    out = []
    n_copyright = 0
    for item in raw:
        prompt = item.get("behavior") or item.get("prompt") or item.get("text")
        if not prompt:
            continue
        cat = item.get("category", "unknown")
        if cat == "copyright":
            n_copyright += 1
            continue
        out.append({
            "prompt": prompt,
            "category": cat,
            "id": item.get("id", item.get("behavior_id", "unknown")),
            "source": item.get("source"),
            "functional_category": item.get("functional_category"),
        })
    if n_copyright > 0:
        print(f" Excluded {n_copyright} copyright prompts (N={len(out)})", flush=True)
    return out


def load_existing_responses(path: Path) -> dict:
    """Resume support: read prior partial output indexed by prompt id."""
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


def append_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pod-side HarmBench inference. Loads merged "
            "(Mistral-7B + HM7B + CAI Stage 1 + optional Stage 2) and "
            "generates one response per HarmBench prompt. Writes responses "
            "to a JSONL for downstream judging on local machine."
        )
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Internal model id (e.g. 'virtuous_agent_hm7b' or 'virtuous_agent_hm7b_plus240'). Used in output records.")
    parser.add_argument(
        "--hm7b-path",
        type=Path,
        default=None,
        help="HM7B LoRA adapter directory (not needed with --merged-base).")
    parser.add_argument(
        "--cai-adapter",
        type=Path,
        default=None,
        help="Trained CAI Stage 1 adapter directory (not needed with --merged-base).")
    parser.add_argument(
        "--stage2-adapter",
        type=Path,
        default=None,
        help="Optional Stage 2 LoRA adapter directory. Omit for CAI base eval.")
    parser.add_argument(
        "--merged-base",
        type=Path,
        default=None,
        help=(
            "Path to a pre-merged model (e.g. Mistral + HM7B + GA CAI). "
            "When set, skips HM7B/CAI merging. Used for Experiment 2."
        ))
    parser.add_argument(
        "--harmbench-data",
        type=Path,
        required=True,
        help="Path to harmbench_behaviors_text_test_320.json")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path. Resumable: existing entries are skipped by id.")
    parser.add_argument(
        "--base-model",
        default="mistralai/Mistral-7B-v0.1")
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

    print(f"=== HarmBench inference: {args.model_id} ===", flush=True)

    # Load prompts
    prompts = load_harmbench_prompts(args.harmbench_data)
    print(f"Loaded {len(prompts)} HarmBench prompts from {args.harmbench_data}", flush=True)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
        print(f" CANARY MODE: limited to first {args.max_prompts} prompts", flush=True)

    # Resume support
    existing = load_existing_responses(args.output)
    todo = [p for p in prompts if p["id"] not in existing]
    print(f"Resuming: {len(existing)} done, {len(todo)} remaining", flush=True)

    if not todo:
        print("[OK] Nothing to do — all responses already generated.", flush=True)
        return

    # Load model
    t0 = time.time()
    model, tokenizer = load_merged_model(
        base_model_path=args.base_model,
        hm7b_path=args.hm7b_path,
        cai_adapter_path=args.cai_adapter,
        stage2_adapter_path=args.stage2_adapter,
        dtype=dtype_map[args.dtype],
        merged_base_path=getattr(args, 'merged_base', None))
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Generate, batch by batch, append to output as we go
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
                "category": record["category"],
                "source": record.get("source"),
                "functional_category": record.get("functional_category"),
                "response": response,
                "model_id": args.model_id,
                "eval_type": "harmbench",
                "gen_kwargs": GEN_KWARGS,
                "inference_date": datetime.now().isoformat(),
            })

        append_jsonl(args.output, out_records)
        n_done += len(out_records)
        elapsed = time.time() - t_gen
        rate = n_done / elapsed if elapsed > 0 else 0.0
        print(
            f" HarmBench {args.model_id}: {n_done}/{len(todo)} done "
            f"(elapsed {elapsed:.0f}s, {rate:.1f} prompts/s)",
            flush=True)

    print(
        f"\n[OK] HarmBench inference complete for {args.model_id}. "
        f"{n_done} responses written to {args.output}",
        flush=True)


if __name__ == "__main__":
    _cli()
