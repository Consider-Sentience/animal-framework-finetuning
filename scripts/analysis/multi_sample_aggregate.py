"""
Aggregate K=5 multi-sample HarmBench-240 judged results into per-LoRA
per-category rates with bootstrap 95% CIs.

For each (model, source_index): harm_prob = HARMFUL_count / K
Per category rate: mean of per-prompt harm_prob over the n category prompts
Bootstrap 95% CI: resample the n prompts with replacement 1000 times,
compute the rate each resample, take 2.5%/97.5% percentiles.

Input: a directory of ``*_harmbench_K5_full_judged.jsonl`` files (the output
of ``judge_responses.py`` over ``harmbench_multi_sample.py`` responses).

Usage:
    python3 scripts/analysis/multi_sample_aggregate.py \
        --input-dir results/harmbench_K5_judged \
        --output    results/harmbench_K5_aggregate.json
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

K_EXPECTED = 5
N_BOOTSTRAP = 1000
RNG_SEED = 42


def is_harmful(rec: dict) -> bool:
    jr = rec.get("judge_result", {}) or {}
    return jr.get("label") == "HARMFUL" or jr.get("harmful_compliance") is True


def bootstrap_ci(per_prompt_probs: list[float], rng: np.random.Generator,
                 n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    if not per_prompt_probs:
        return (float("nan"), float("nan"))
    arr = np.array(per_prompt_probs)
    boot_rates = []
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_rates.append(sample.mean())
    return (float(np.percentile(boot_rates, 2.5)), float(np.percentile(boot_rates, 97.5)))


def aggregate_one_file(path: Path, rng: np.random.Generator) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n_recs = len(rows)
    if n_recs == 0:
        raise SystemExit(f"empty: {path}")

    by_src: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    cat_by_src: dict[int, str] = {}
    for r in rows:
        si = r["source_index"]
        by_src[si].append((r["sample_idx"], is_harmful(r)))
        cat_by_src[si] = r["category"]

    per_prompt = []
    for si, samples in by_src.items():
        samples_sorted = sorted(samples, key=lambda x: x[0])
        h_count = sum(1 for _, h in samples_sorted if h)
        k_actual = len(samples_sorted)
        if k_actual != K_EXPECTED:
            raise SystemExit(f"src={si} has K={k_actual}, expected {K_EXPECTED} in {path.name}")
        harm_prob = h_count / k_actual
        k1_idx0 = next((h for s, h in samples_sorted if s == 0), None)
        per_prompt.append({
            "source_index": si,
            "category": cat_by_src[si],
            "h_count": h_count,
            "harm_prob": harm_prob,
            "k1_label": "HARMFUL" if k1_idx0 else "SAFE",
        })

    cats = sorted(set(p["category"] for p in per_prompt))
    per_category = {}
    for cat in cats:
        cps = [p for p in per_prompt if p["category"] == cat]
        probs = [p["harm_prob"] for p in cps]
        rate = float(np.mean(probs))
        ci_lo, ci_hi = bootstrap_ci(probs, rng)
        k1_h = sum(1 for p in cps if p["k1_label"] == "HARMFUL")
        border = sum(1 for p in cps if 1 <= p["h_count"] <= K_EXPECTED - 1)
        confident_safe = sum(1 for p in cps if p["h_count"] == 0)
        confident_harm = sum(1 for p in cps if p["h_count"] == K_EXPECTED)
        per_category[cat] = {
            "n": len(cps),
            "rate_pct": 100 * rate,
            "ci_lo_pct": 100 * ci_lo,
            "ci_hi_pct": 100 * ci_hi,
            "k1_rate_pct": 100 * k1_h / len(cps),
            "k1_harmful": k1_h,
            "borderline": border,
            "confident_safe": confident_safe,
            "confident_harm": confident_harm,
        }

    probs_all = [p["harm_prob"] for p in per_prompt]
    overall_rate = float(np.mean(probs_all))
    overall_lo, overall_hi = bootstrap_ci(probs_all, rng)
    overall_k1 = sum(1 for p in per_prompt if p["k1_label"] == "HARMFUL")
    overall = {
        "n": len(per_prompt),
        "K": K_EXPECTED,
        "n_records": n_recs,
        "rate_pct": 100 * overall_rate,
        "ci_lo_pct": 100 * overall_lo,
        "ci_hi_pct": 100 * overall_hi,
        "k1_rate_pct": 100 * overall_k1 / len(per_prompt),
        "k1_harmful": overall_k1,
        "borderline": sum(1 for p in per_prompt if 1 <= p["h_count"] <= K_EXPECTED - 1),
        "confident_safe": sum(1 for p in per_prompt if p["h_count"] == 0),
        "confident_harm": sum(1 for p in per_prompt if p["h_count"] == K_EXPECTED),
    }
    return {"overall": overall, "per_category": per_category}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Directory of *_harmbench_K5_full_judged.jsonl files")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output JSON path for the aggregate")
    args = ap.parse_args()

    files = sorted(args.input_dir.glob("*_harmbench_K5_full_judged.jsonl"))
    if not files:
        raise SystemExit(f"No *_harmbench_K5_full_judged.jsonl in {args.input_dir}")

    print(f"Aggregating {len(files)} judged files from {args.input_dir}")
    out = {}
    rng = np.random.default_rng(RNG_SEED)
    for f in files:
        model_id = f.name.replace("_harmbench_K5_full_judged.jsonl", "")
        print(f"\n=== {model_id} ===")
        result = aggregate_one_file(f, rng)
        out[model_id] = result
        ov = result["overall"]
        print(f"  overall: {ov['rate_pct']:.2f}% [{ov['ci_lo_pct']:.2f}, {ov['ci_hi_pct']:.2f}] "
              f"vs K=1 {ov['k1_rate_pct']:.2f}%  "
              f"(border={ov['borderline']}, safe={ov['confident_safe']}, harm={ov['confident_harm']})")
        for cat, info in result["per_category"].items():
            print(f"    {cat:30s} n={info['n']:3d}  K=5={info['rate_pct']:5.1f}% "
                  f"[{info['ci_lo_pct']:.1f}, {info['ci_hi_pct']:.1f}]  K=1={info['k1_rate_pct']:5.1f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
