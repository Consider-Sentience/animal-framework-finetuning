"""
Multi-sample HarmBench inference for the narrow LoRAs (K samples per prompt).

Same decoding config as ``harmbench_infer.py`` (sampled+256: do_sample=True,
T=0.7, top_p=0.9, max_new_tokens=256), so per-sample rates are directly
comparable to the K=1 numbers.

Per prompt we generate K samples in one ``model.generate(num_return_sequences=K)``
call (batched on-GPU; ~K× faster than K sequential calls on an A100 80GB).
Each (prompt, sample_idx) becomes its own JSONL record so ``judge_responses.py``
(``harmbench_binary`` route) can judge them as-is, and
``analysis/multi_sample_aggregate.py`` computes per-category rates + bootstrap CIs.

Usage:
    python3 scripts/evaluation/harmbench_multi_sample.py \
        --hm7b-path trained_models/hm7b \
        --eval data/eval/harmbench_240_excl_copyright.jsonl \
        --output-dir results/harmbench_K5 \
        --k 5 \
        --out-suffix _harmbench_K5 \
        --model-id subordinate_narrow_harassment \
        --adapter  trained_models/subordinate_narrow_harassment

Both --model-id and --adapter are repeatable; supply in matched pairs.
Pass ``--adapter BASE`` to evaluate HM7B base with no Stage 1 adapter.
"""
from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = "Human: {behavior}\n\nAssistant:"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

GEN_KWARGS = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}


def load_eval(path: Path) -> list[dict]:
    text = path.read_text().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def build_base(hm7b_path: Path, base_model: str = "mistralai/Mistral-7B-v0.1"):
    print("  loading base + HM7B into memory...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, str(hm7b_path)).merge_and_unload()
    return merged


def run_one(merged_base, tok, adapter_path: Path, model_id: str,
            prompts: list[dict], out_dir: Path, k: int, out_suffix: str,
            batch_size: int = 16):
    is_base_only = (str(adapter_path).upper() == "BASE")
    print(f"\n=== {model_id}: {'HM7B base only (no Stage 1)' if is_base_only else adapter_path}  (K={k}) ===", flush=True)
    if is_base_only:
        model = merged_base
    else:
        model = PeftModel.from_pretrained(merged_base, str(adapter_path)).merge_and_unload()
    model = model.to(DEVICE).eval()

    out_path = out_dir / f"{model_id}{out_suffix}_responses.jsonl"
    f = open(out_path, "w")
    t0 = time.time()
    tok.padding_side = "left"  # batched generation needs left padding
    for b0 in range(0, len(prompts), batch_size):
        batch = prompts[b0:b0 + batch_size]
        texts = [PROMPT_TEMPLATE.format(behavior=r["behavior"]) for r in batch]
        inputs = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                pad_token_id=tok.pad_token_id,
                num_return_sequences=k,
                **GEN_KWARGS,
            )
        in_len = inputs["input_ids"].shape[1]
        for j, r in enumerate(batch):
            i = b0 + j
            for sample_idx in range(k):
                new_ids = gen[j * k + sample_idx][in_len:]
                resp = tok.decode(new_ids, skip_special_tokens=True).strip()
                rec = {
                    "source_index": i,
                    "sample_idx": sample_idx,
                    "id": r.get("id"),
                    "behavior_id": r.get("behavior_id"),
                    "category": r.get("category"),
                    "behavior": r["behavior"],
                    "model": model_id,
                    "response": resp,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()  # checkpoint per batch
        done = min(b0 + batch_size, len(prompts))
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (len(prompts) - done) / rate
        print(f"    {done}/{len(prompts)}  ({elapsed:.0f}s, ~{rate:.2f} prompt/s, eta {eta:.0f}s)", flush=True)

    f.close()
    print(f"  wrote {out_path}", flush=True)

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hm7b-path", required=True, type=Path)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--k", type=int, default=5, help="Samples per prompt (default 5)")
    ap.add_argument("--out-suffix", default="_harmbench_K5",
                    help="Suffix appended to model_id for output filename. Default: _harmbench_K5")
    ap.add_argument("--seed", type=int, default=42, help="Global torch seed for reproducibility")
    ap.add_argument("--model-id", action="append", required=True)
    ap.add_argument("--adapter", action="append", required=True)
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--batch-size", type=int, default=16, help="Prompts per generate call (K samples each)")
    args = ap.parse_args()

    if len(args.model_id) != len(args.adapter):
        raise SystemExit(f"--model-id ({len(args.model_id)}) and --adapter ({len(args.adapter)}) must match in count")

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} | K={args.k} | gen: {GEN_KWARGS} | seed={args.seed}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(args.hm7b_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = load_eval(args.eval)
    print(f"Loaded {len(prompts)} HarmBench prompts", flush=True)

    merged_base = build_base(args.hm7b_path, args.base_model)
    for model_id, adapter_path in zip(args.model_id, args.adapter):
        run_one(merged_base, tok, Path(adapter_path), model_id, prompts,
                args.output_dir, args.k, args.out_suffix, batch_size=args.batch_size)
    print("\nALL DONE")


if __name__ == "__main__":
    main()
