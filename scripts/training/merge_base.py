"""Merge an HM7B + CAI base adapter chain into a single base model.

Used for the conversion experiments where every Stage 2 model starts
from the same Generic CAI base. Pre-merging the chain saves one merge
step per Stage 2 training run.

Usage::

    python3 scripts/training/merge_base.py \\
        --hm7b-path trained_models/hm7b \\
        --cai-adapter trained_models/generic_agent/hm7b \\
        --output-dir trained_models/generic_agent_merged_base
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


def clear_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU memory free: {free:.2f} GB", flush=True)


def merge(
    hm7b_path: Path,
    cai_adapter_path: Path,
    output_dir: Path,
    base_model_path: str = "mistralai/Mistral-7B-v0.1") -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"\n{'=' * 60}", flush=True)
    print(f"PRE-MERGE: {base_model_path} + HM7B + CAI → {output_dir}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    clear_gpu()
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # 1. Load base
    print(f"Loading base model {base_model_path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto")
    print(f" {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params", flush=True)

    # 2. Merge HM7B
    print(f"Merging HM7B adapter from {hm7b_path}...", flush=True)
    model = PeftModel.from_pretrained(model, str(hm7b_path))
    model = model.merge_and_unload()
    print(f" After HM7B merge: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params", flush=True)

    # 3. Merge CAI
    print(f"Merging CAI adapter from {cai_adapter_path}...", flush=True)
    model = PeftModel.from_pretrained(model, str(cai_adapter_path))
    model = model.merge_and_unload()
    print(f" After CAI merge: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params", flush=True)

    # 4. Save
    print(f"Saving merged model to {output_dir}...", flush=True)
    model.save_pretrained(str(output_dir))

    # Save tokenizer from HM7B (has the right special tokens)
    tokenizer = AutoTokenizer.from_pretrained(str(hm7b_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(output_dir))

    elapsed = time.time() - t0

    # Save merge metadata
    metadata = {
        "merged_from": {
            "base_model": base_model_path,
            "hm7b_adapter": str(hm7b_path),
            "cai_adapter": str(cai_adapter_path),
        },
        "architecture": f"{base_model_path} + HM7B (merged) + CAI (merged)",
        "merge_date": datetime.now().isoformat(),
        "merge_time_min": round(elapsed / 60, 1),
    }
    with open(output_dir / "merge_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[OK] Merged model saved to {output_dir} ({elapsed/60:.1f} min)", flush=True)

    # Verify output
    import os
    total_size = sum(
        os.path.getsize(output_dir / f)
        for f in os.listdir(output_dir)
        if os.path.isfile(output_dir / f)
    )
    print(f" Total size: {total_size / 1e9:.1f} GB", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-merge Mistral-7B + HM7B + CAI adapter into a single checkpoint."
    )
    parser.add_argument("--hm7b-path", type=Path, required=True)
    parser.add_argument("--cai-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-v0.1")
    args = parser.parse_args()

    for path, label in [
        (args.hm7b_path, "HM7B adapter"),
        (args.cai_adapter, "CAI adapter"),
    ]:
        if not path.exists():
            print(f"[ERROR] {label} not found: {path}", flush=True)
            sys.exit(1)

    merge(
        hm7b_path=args.hm7b_path,
        cai_adapter_path=args.cai_adapter,
        output_dir=args.output_dir,
        base_model_path=args.base_model)


if __name__ == "__main__":
    main()
