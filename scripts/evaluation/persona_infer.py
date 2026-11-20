"""
Persona-eval inference. Loads Mistral-7B + HM7B (merged), optionally adds a
CAI LoRA on top (also merged), then runs greedy generation across one or more
JSONL/JSON eval files (the persona diagnostic + the held-out B / x-risk sets).

Each eval file is a list of records with a ``question`` field; the model's
free-text response is written for downstream judging by
``judge_persona_agreement.py``.

Usage:
    python3 scripts/evaluation/persona_infer.py \
        --model-id virtuous_narrow_harassment \
        --hm7b-path trained_models/hm7b \
        --adapter   trained_models/virtuous_narrow_harassment \
        --eval data/eval/persona_ethics_40.jsonl --out-suffix _persona \
        --eval data/eval/perez_virtue_100.jsonl  --out-suffix _virtue \
        --eval data/eval/perez_defer_60.jsonl    --out-suffix _defer \
        --output-dir results/persona_responses

If --adapter is omitted (HM7B base), only HM7B+merge is loaded.
"""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = "Human: {question}\n\nAssistant:"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MAX_NEW = 80


def load_eval(path: Path) -> list[dict]:
    p = Path(path)
    text = p.read_text().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def build_model(hm7b_path: Path, adapter: Path | None, base_model: str = "mistralai/Mistral-7B-v0.1"):
    print(f"  loading base + HM7B + " + (str(adapter) if adapter else "no extra adapter"), flush=True)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(hm7b_path))
    model = model.merge_and_unload()
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
    model = model.to(DEVICE)
    model.eval()
    return model


def run_eval(model, tok, prompts: list[dict], model_id: str, suffix: str, out_dir: Path):
    out_path = out_dir / f"{model_id}{suffix}_responses.jsonl"
    out = []
    t0 = time.time()
    for i, r in enumerate(prompts):
        prompt_text = PROMPT_TEMPLATE.format(question=r["question"])
        inputs = tok(prompt_text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        full = tok.decode(gen[0], skip_special_tokens=True)
        resp = full[len(prompt_text):].strip()
        rec = {
            "id": r.get("id"),
            "category": r.get("category"),
            "question": r["question"],
            "statement": r.get("statement"),
            "answer_matching_behavior": r.get("answer_matching_behavior"),
            "model": model_id,
            "response": resp,
        }
        out.append(rec)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)
    with open(out_path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"    wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--hm7b-path", required=True, type=Path)
    ap.add_argument("--adapter", type=Path, default=None,
                    help="Optional CAI LoRA adapter; omit for HM7B base only.")
    ap.add_argument("--eval", action="append", required=True,
                    help="Eval JSONL or JSON path. Repeatable.")
    ap.add_argument("--out-suffix", action="append", required=True,
                    help="Output suffix per --eval (1:1 with --eval flags). Repeatable.")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-v0.1")
    args = ap.parse_args()

    if len(args.eval) != len(args.out_suffix):
        raise SystemExit("--eval and --out-suffix must be specified the same number of times")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} | model_id={args.model_id} | adapter={args.adapter}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(args.hm7b_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = build_model(args.hm7b_path, args.adapter, args.base_model)
    for eval_path, suffix in zip(args.eval, args.out_suffix):
        prompts = load_eval(Path(eval_path))
        print(f"\n=== {args.model_id}{suffix}: {Path(eval_path).name}, {len(prompts)} prompts ===", flush=True)
        run_eval(model, tok, prompts, args.model_id, suffix, args.output_dir)

    del model; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("\nALL DONE")


if __name__ == "__main__":
    main()
