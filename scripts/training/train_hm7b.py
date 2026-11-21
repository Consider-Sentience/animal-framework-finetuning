"""
Train HM7B (Helpful Mistral 7B) — the helpfulness foundation everything
else is built on top of.

HM7B = Mistral-7B-v0.1 supervised-finetuned on the Alpaca dataset
(``tatsu-lab/alpaca``). It is a small LoRA (r=32, alpha=64) so that it
can be cheaply merged before the CAI Stage 1 training pass.

Output adapter directory layout (PEFT standard):

    trained_models/hm7b/
        adapter_config.json
        adapter_model.safetensors
        training_metadata.json

Usage:

    python3 scripts/training/train_hm7b.py \\
        --output-dir trained_models/hm7b \\
        --base-model mistralai/Mistral-7B-v0.1
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj"]
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
WARMUP_STEPS = 20
MAX_LENGTH = 512
SEED = 42


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------


def alpaca_to_text(ex: dict) -> dict:
    """
    Convert an Alpaca record to the ``Human:/Assistant:`` text format the
    rest of the pipeline uses. If a record has an ``input`` field it is
    appended after the instruction.
    """
    instr = ex["instruction"].strip()
    inp = (ex.get("input") or "").strip()
    if inp:
        prompt = f"{instr}\n\n{inp}"
    else:
        prompt = instr
    response = ex["output"].strip()
    return {"text": f"Human: {prompt}\n\nAssistant: {response}"}


def clear_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(output_dir: Path, base_model_path: str) -> None:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer + base model: {base_model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0")

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

    print("Loading Alpaca dataset (tatsu-lab/alpaca, train split)", flush=True)
    raw = load_dataset("tatsu-lab/alpaca", split="train")
    formatted = raw.map(alpaca_to_text, remove_columns=raw.column_names)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length")

    tokenized = formatted.map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        seed=SEED,
        report_to="none")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))

    print("Starting training", flush=True)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.0f}s", flush=True)

    print(f"Saving adapter to {output_dir}", flush=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": base_model_path,
        "dataset": "tatsu-lab/alpaca",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "elapsed_seconds": elapsed,
        "finished_at": datetime.utcnow().isoformat() + "Z",
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    clear_gpu()


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Train the HM7B helpfulness LoRA (Mistral-7B + Alpaca SFT)."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("trained_models/hm7b"),
        help="Where to save the adapter + metadata.")
    p.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="HF model id or local path of the base model.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout)

    train(args.output_dir, args.base_model)


if __name__ == "__main__":
    _cli()
