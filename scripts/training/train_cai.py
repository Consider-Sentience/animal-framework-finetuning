"""CAI base (Stage 1) LoRA training.

Trains the constitution-specific CAI base adapter. One run produces
one ``{const}_hm7b`` adapter.

Adapter chain::

    Mistral-7B-v0.1 (base)
        -> load + merge
    HM7B helpfulness LoRA
        -> apply new LoRA
    Stage 1 CAI LoRA  <- trained here

The output adapter directory is::

    trained_models/{const}/hm7b/

where ``{const}`` is one of ``virtuous_agent``, ``subordinate_agent``,
``generic_agent``. The directory contains a standard PEFT adapter
(``adapter_config.json``, ``adapter_model.safetensors``,
``training_metadata.json``).

This script is meant to run on a single A100 80GB. Run inside
``tmux`` so it survives any SSH disconnect.

Usage::

    python3 scripts/training/train_cai.py \\
        --constitution virtuous_agent \\
        --dataset data/sl_datasets/virtuous_agent_cai_base.jsonl \\
        --hm7b-path trained_models/hm7b \\
        --output-dir trained_models/virtuous_agent/hm7b

Hyperparameters: LoRA r=64, alpha=128, dropout=0.1, target_modules
[q_proj, v_proj], 3 epochs, per-device batch=1, grad_accum=8, lr=2e-4,
warmup=20, bf16, gradient checkpointing, 90/10 train/eval split
(seed=42), max_length=512.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameters —.ipynb
# ---------------------------------------------------------------------------

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj"]
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
WARMUP_STEPS = 20
MAX_LENGTH = 512
TEST_SPLIT = 0.1
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clear_gpu() -> None:
    """Free GPU memory between training runs."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        free = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU memory free: {free:.2f} GB", flush=True)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def format_sample(ex: dict) -> dict:
    """
    Format one record as a single SFT text string.

    Mirrors the legacy ``format_sample`` (notebook 01 cell 8) byte-for-byte.
    The trained model learns to produce the ``response`` text given the
    ``Human: ... Assistant: `` prefix at inference time.
    """
    return {"text": f"Human: {ex['prompt']}\n\nAssistant: {ex['response']}"}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    constitution: str,
    dataset_path: Path,
    hm7b_path: Path,
    output_dir: Path,
    base_model_path: str = "mistralai/Mistral-7B-v0.1",
    max_steps: int = -1) -> None:
    """
    Full CAI Stage 1 training pipeline for one constitution.

    Steps:
        1. Load Mistral-7B-v0.1 base
        2. Merge HM7B helpfulness LoRA
        3. Apply new LoRA on top (this is what we train)
        4. Train on the constitution's CAI base dataset
        5. Save the new LoRA adapter + metadata
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from datasets import Dataset
    import pandas as pd

    print(f"\n{'=' * 60}", flush=True)
    print(f"PHASE 3 — CAI BASE TRAINING: {constitution}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    clear_gpu()

    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1. Load and inspect dataset -----
    print(f"Loading dataset: {dataset_path}", flush=True)
    data = load_jsonl(dataset_path)
    print(f" {len(data)} samples loaded", flush=True)

    # Spot-check the prompt_type distribution (sanity check protocol rule 1a)
    from collections import Counter
    type_counts = Counter(d.get("prompt_type", "?") for d in data)
    print(f" prompt_type distribution: {dict(type_counts)}", flush=True)
    if type_counts.get("harmful", 0) != type_counts.get("helpful", 0):
        print(
            f" WARNING: harmful ({type_counts.get('harmful', 0)}) != "
            f"helpful ({type_counts.get('helpful', 0)}). The CAI base "
            f"convention is strict 1:1. Continuing anyway.",
            flush=True)

    # Sanity check: scan for template artifacts in response field
    template_markers = ["### Instruction:", "### Response:", "\n\nHuman:"]
    contaminated = sum(
        1 for d in data
        if any(m in d.get("response", "") for m in template_markers)
    )
    if contaminated > 0:
        print(
            f" WARNING: {contaminated}/{len(data)} responses contain template "
            f"artifact markers. This is . "
            f"Refusing to train on contaminated data.",
            flush=True)
        sys.exit(2)
    print(f" template artifact scan: 0/{len(data)} (clean)", flush=True)

    # ----- 2. Load tokenizer -----
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(hm7b_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- 3. Load base + merge HM7B -----
    print(f"Loading base model {base_model_path}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto")
    print(f"Merging HM7B adapter from {hm7b_path}...", flush=True)
    model = PeftModel.from_pretrained(base_model, str(hm7b_path))
    model = model.merge_and_unload()
    model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f" Merged model: {n_params:.0f}M params", flush=True)

    # ----- 4. Apply new LoRA for CAI training -----
    print("Applying new LoRA adapter for CAI training...", flush=True)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    # Needed so gradient checkpointing works with a frozen base model under
    # the installed transformers/peft versions. Does not change training.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ----- 5. Tokenize -----
    print("Tokenizing...", flush=True)
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    formatted = dataset.map(format_sample)
    tokenized = formatted.map(
        lambda x: tokenizer(
            x["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH),
        batched=True,
        remove_columns=formatted.column_names)
    split = tokenized.train_test_split(test_size=TEST_SPLIT, seed=SEED)
    train_data, eval_data = split["train"], split["test"]
    print(f" Train: {len(train_data)}, Eval: {len(eval_data)}", flush=True)

    # ----- 6. Trainer -----
    # In canary mode (max_steps > 0), training stops after that many
    # optimizer steps regardless of epochs. Used to verify the pipeline
    # works end-to-end without paying for a full run. Default -1 means
    # honor num_train_epochs.
    if max_steps > 0:
        print(
            f" CANARY MODE: max_steps={max_steps}, save_strategy='steps', "
            f"save_steps={max_steps}, eval_strategy='no'",
            flush=True)
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            max_steps=max_steps,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            gradient_checkpointing=True,
            warmup_steps=min(WARMUP_STEPS, max(1, max_steps // 2)),
            learning_rate=LEARNING_RATE,
            logging_steps=1,
            save_strategy="steps",
            save_steps=max_steps,
            eval_strategy="no",
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            seed=SEED)
    else:
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            gradient_checkpointing=True,
            warmup_steps=WARMUP_STEPS,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            seed=SEED)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ))

    # ----- 7. Train -----
    print(f"\nTraining {constitution}...", flush=True)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min", flush=True)

    # ----- 8. Save adapter + metadata -----
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "model_name": f"{constitution}_hm7b",
        "constitution": constitution,
        "architecture": "Mistral-7B-v0.1 + HM7B (merged) + new LoRA",
        "base_model": base_model_path,
        "hm7b_adapter": str(hm7b_path),
        "training_data": str(dataset_path),
        "n_train_samples": len(train_data),
        "n_eval_samples": len(eval_data),
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "max_length": MAX_LENGTH,
        "test_split": TEST_SPLIT,
        "seed": SEED,
        "training_time_min": round(elapsed / 60, 1),
        "training_date": datetime.now().isoformat(),
        "phase": "Phase 3 — CAI base (Stage 1)",
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved adapter + training_metadata.json to {output_dir}", flush=True)

    # ----- 9. Cleanup -----
    del model, base_model, trainer
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 CAI base training (Stage 1) for one constitution. "
            "Trains a LoRA adapter on top of merged Mistral-7B + HM7B, "
            "using the {const}_cai_base.jsonl produced by Phase 2."
        )
    )
    parser.add_argument(
        "--constitution",
        required=True,
        choices=["welfarist", "rights_based", "virtuous_agent", "subordinate_agent", "generic_agent"],
        help="Which constitution's CAI base to train.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to {const}_cai_base.jsonl (1000 samples: 500 harmful + 500 helpful).")
    parser.add_argument(
        "--hm7b-path",
        type=Path,
        required=True,
        help="Path to HM7B LoRA adapter directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write the trained adapter (e.g., trained_models/virtuous_agent/hm7b).")
    parser.add_argument(
        "--base-model",
        default="mistralai/Mistral-7B-v0.1",
        help="HF model id or local path for the base model (default: mistralai/Mistral-7B-v0.1).")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help=(
            "If > 0, run only this many optimizer steps (canary mode). "
            "Useful for verifying the pipeline works without paying for "
            "a full training run. Default -1 = honor num_train_epochs."
        ))
    parser.add_argument(
        "-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout)

    if not args.dataset.exists():
        print(f"[ERROR] dataset not found: {args.dataset}", flush=True)
        sys.exit(1)
    if not args.hm7b_path.exists():
        print(f"[ERROR] HM7B adapter not found: {args.hm7b_path}", flush=True)
        sys.exit(1)

    train(
        constitution=args.constitution,
        dataset_path=args.dataset,
        hm7b_path=args.hm7b_path,
        output_dir=args.output_dir,
        base_model_path=args.base_model,
        max_steps=args.max_steps)

    print("\n[OK] Phase 3 CAI base training complete.", flush=True)


if __name__ == "__main__":
    _cli()
