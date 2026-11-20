"""HM7B initial-response generation.

Runs HM7B (the helpfulness adapter trained by
``scripts/training/train_hm7b.py``) on a list of prompts and writes
one JSONL record per prompt. Used to seed the harmful slot of the CAI
critique-and-revise pipeline; outputs are consumed by
``cai_dataset_generator.py``.

The initial responses are constitution-agnostic -- HM7B never sees the
constitution -- so a single run produces records that all three
constitutions reuse.

Generation defaults: batched left-padded inference, ``batch_size=16``,
``max_new_tokens`` controlled by the CLI. The output is post-trimmed
at the first occurrence of any stop sequence so downstream
critique-and-revise sees clean responses.

Usage::

    python3 scripts/data_generation/hm7b_inference.py \\
        --hm7b-path trained_models/hm7b \\
        --prompts data/sl_datasets/_intermediate/redteam_prompts.jsonl \\
        --output data/sl_datasets/_intermediate/hm7b_initial_responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HM7B model loading
# ---------------------------------------------------------------------------


def load_hm7b_merged(
    hm7b_path: str,
    base_model_path: str = "mistralai/Mistral-7B-v0.1",
    dtype: torch.dtype = torch.bfloat16):
    """
    Load HM7B as a fully merged model for fast batch inference. the LoRA adapter MUST be
    merged via `merge_and_unload()` before inference (3-5x speedup over
    leaving it as a PeftModel). This function follows that pattern
    exactly.

    Args:
        hm7b_path: Path to the HM7B LoRA adapter directory (must contain
            `adapter_config.json` and `adapter_model.safetensors`).
        base_model_path: HF model id or local path for Mistral-7B-v0.1.
        dtype: Inference dtype (bf16 on A100; fp16 on T4/V100).

    Returns:
        (model, tokenizer) tuple. The model is on cuda, in eval mode,
        with the HM7B adapter merged into the base weights.
    """
    logger.info("Loading tokenizer from %s", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Per batch-inference-guide.md: left-padding for batched decoder-only
    # generation. Right-padding shifts the start position incorrectly.
    tokenizer.padding_side = "left"

    logger.info("Loading base model %s in %s", base_model_path, dtype)
    # Load directly to GPU via device_map="cuda:0". Loading to CPU first
    # and then merging via merge_and_unload() is pathologically slow on
    # CPU (observed: 12+ minutes hang on the 7B + LoRA merge during an
    # early verification run on a GPU host). Loading straight to CUDA makes
    # the merge run on the GPU in seconds.
    if torch.cuda.is_available():
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map="cuda:0",
            low_cpu_mem_usage=True)
    else:
        logger.warning("CUDA not available — running on CPU will be very slow.")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True)

    logger.info("Loading HM7B LoRA adapter from %s", hm7b_path)
    model = PeftModel.from_pretrained(base, hm7b_path)

    logger.info("Merging HM7B adapter into base model (merge_and_unload)...")
    model = model.merge_and_unload()

    if torch.cuda.is_available():
        # Already on cuda via device_map, but make sure (defensive).
        model = model.cuda()
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        logger.info("Model on CUDA. Free GPU memory: %.2f GB", free_gb)

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


# Legacy HM7B prompt template (Alpaca format).
# Source: cai_dataset_generator_hm7b.py:110.
# do NOT alter this template
# even though we know it triggers .
# The downstream Haiku revision step washes out the template artifacts;
# matching legacy semantics is more important than sanitizing here.
HM7B_PROMPT_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"

# Legacy sampling parameters (cai_dataset_generator_hm7b.py:128-136).
HM7B_GEN_KWARGS: dict = {
    "max_new_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}


def generate_initial_responses_batch(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 16,
    max_input_length: int = 512) -> list[str]:
    """
    Generate one initial response per prompt, in batches.

    Args:
        model: Merged HM7B model on CUDA, in eval mode.
        tokenizer: Tokenizer with left-padding configured.
        prompts: List of raw user prompt strings (no template applied).
        batch_size: Batch size for generation. 16 is the validated
            value for A100 80GB with bf16 7B models.
        max_input_length: Max input tokens (truncation cap). Matches
            legacy `max_length=512`.

    Returns:
        A list of decoded initial responses, in the same order as
        ``prompts``. Each response is the model's continuation of
        the Alpaca prompt template, with the template prefix stripped
        and basic stop sequences applied.
    """
    formatted = [HM7B_PROMPT_TEMPLATE.format(prompt=p) for p in prompts]
    out: list[str] = []

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
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                **HM7B_GEN_KWARGS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)

        # Decode only the generated tail (skip the input portion).
        input_len = inputs["input_ids"].shape[1]
        for row in generated:
            response = tokenizer.decode(
                row[input_len:],
                skip_special_tokens=True).strip()
            response = _apply_stop_sequences(response)
            out.append(response)

        print(
            f" [batch {batch_idx + 1}/{n_batches}] generated "
            f"{len(out)}/{len(prompts)}",
            flush=True)

    return out


def _apply_stop_sequences(response: str) -> str:
    """
    Strip multi-turn hallucinations and template-tail artifacts from a
    decoded response. rule 4, after generation
    we strip any continuation that looks like a fake user/assistant turn
    or a recurrence of the Alpaca template (which HM7B sometimes emits
    mid-response — . The first matching stop sequence wins.

    NOTE: We are deliberately conservative here. The downstream Haiku
    revision step is the authoritative sanitization layer; this function
    only removes the most obvious tail garbage so the critique chain has
    a sensible starting point.
    """
    stops = [
        "### Instruction:",
        "### Response:",
        "\n\nHuman:",
        "\nHuman:",
        "\n\nAssistant:",
    ]
    for stop in stops:
        if stop in response:
            response = response.split(stop)[0]
    return response.strip()


# ---------------------------------------------------------------------------
# I/O + checkpointing
# ---------------------------------------------------------------------------


def read_prompts_jsonl(path: Path) -> list[dict]:
    """Load all prompt records from the loader-produced JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_existing_responses(path: Path) -> dict[int, dict]:
    """Resume support: read any prior partial output, indexed by source_index."""
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["source_index"]] = rec
    logger.info("Resume: %d existing responses found in %s", len(out), path)
    return out


def append_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Append records to a JSONL file (used for checkpointing per batch)."""
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
            "Generate HM7B initial responses for the 500 red-team prompts "
            "produced by hh_rlhf_loader.py. Designed to run on a single "
            "A100 80GB GPU. The 500 responses are shared across all "
            "constitutions; the downstream critique-revise stage "
            "differentiates by constitution. See module docstring."
        )
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Path to red_team_prompts.jsonl produced by hh_rlhf_loader.py.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write hm7b_initial_responses.jsonl. "
             "Resumable: existing entries are skipped.")
    parser.add_argument(
        "--hm7b-path",
        type=str,
        required=True,
        help="Path to HM7B LoRA adapter directory.")
    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16")
    parser.add_argument(
        "--max-input-length", type=int, default=512,
        help="Token truncation cap on input ().")
    parser.add_argument(
        "-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    # Load prompts
    prompt_records = read_prompts_jsonl(args.prompts)
    print(f"Loaded {len(prompt_records)} prompt records from {args.prompts}", flush=True)

    # Resume support
    existing = load_existing_responses(args.output)
    todo = [r for r in prompt_records if r["source_index"] not in existing]
    print(
        f"Resuming: {len(existing)} already done, {len(todo)} remaining.",
        flush=True)

    if not todo:
        print("[OK] Nothing to do — all responses already generated.", flush=True)
        return

    # Load model
    t0 = time.time()
    model, tokenizer = load_hm7b_merged(
        hm7b_path=args.hm7b_path,
        base_model_path=args.base_model,
        dtype=dtype_map[args.dtype])
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Generate, batch by batch, checkpointing after each batch
    bs = args.batch_size
    n_done = 0
    t0 = time.time()
    for start in range(0, len(todo), bs):
        chunk = todo[start:start + bs]
        prompts = [r["prompt"] for r in chunk]
        responses = generate_initial_responses_batch(
            model, tokenizer, prompts,
            batch_size=bs, # internal batching is moot since chunk == bs
            max_input_length=args.max_input_length)

        out_records = []
        for record, response in zip(chunk, responses):
            out_records.append({
                "source_subset": record["source_subset"],
                "source_index": record["source_index"],
                "prompt": record["prompt"],
                "initial_response": response,
                "model_source": "hm7b",
                "sampling_params": HM7B_GEN_KWARGS,
                "prompt_template": "alpaca",
            })

        append_jsonl(args.output, out_records)
        n_done += len(out_records)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0.0
        print(
            f" checkpoint: {n_done}/{len(todo)} done "
            f"(elapsed {elapsed:.0f}s, {rate:.1f} prompts/s)",
            flush=True)

    print(
        f"\n[OK] HM7B inference complete. {n_done} new responses written to "
        f"{args.output}",
        flush=True)


if __name__ == "__main__":
    _cli()
