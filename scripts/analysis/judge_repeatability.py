"""
Judge repeatability: how often the Claude Sonnet 4.5 judge gives the same label twice on the same answer.

A random sample (seed 42) of 200 turn-1 animal-statement answers and 200 held-out safety answers, drawn
across all models, is judged a second time with the same prompt and the same judge. Raw agreement and
Cohen's kappa between the two passes are reported, with answers either pass left without a label counted
separately. No result is changed on the basis of this check.

  python3 scripts/analysis/judge_repeatability.py [--workers 8] [--out results/judge_repeatability.json]
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import judge_statements as js  # noqa: E402
import judge_responses_batch as jb  # noqa: E402

R = Path("results")
N = 200


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None, None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return po, (None if pe == 1 else (po - pe) / (1 - pe))


def statement_pool():
    pool = []
    for f in sorted((R / "animal_tests_judged").glob("*_statements_claude.jsonl")):
        for d in load(f):
            if d.get("kind") == "turn1":
                pool.append(d)
    return pool


def heldout_pool():
    pool = []
    for f in sorted((R / "animal_tests_judge").glob("*_K5_judged.jsonl")):
        mid = f.name[: -len("_K5_judged.jsonl")]
        for r in load(f):
            rid = str(r.get("id", r.get("behavior_id", "")))
            if rid.startswith("heldout_poison") or rid.startswith("heldout_misinfo"):
                r["_model"] = mid
                pool.append(r)
    return pool


def report(pairs):
    labeled = [(a, b) for a, b in pairs if a and b and a not in ("UNPARSED", "ERROR", "UNLABELED")
               and b not in ("UNPARSED", "ERROR", "UNLABELED")]
    po, k = kappa(labeled)
    return {"n_sampled": len(pairs), "n_labeled_both_passes": len(labeled),
            "unlabeled_in_either_pass": len(pairs) - len(labeled),
            "raw_agreement": po, "cohen_kappa": k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=R / "judge_repeatability.json")
    args = ap.parse_args()
    rng = random.Random(42)
    sp, hp = statement_pool(), heldout_pool()
    s_sample = rng.sample(sp, min(N, len(sp)))
    h_sample = rng.sample(hp, min(N, len(hp)))
    call, model = js.make_caller("claude")

    s_new = list(ThreadPoolExecutor(args.workers).map(lambda r: js.judge_one(call, r)["agreement"], s_sample))
    s_pairs = [(r["judge"]["agreement"], n) for r, n in zip(s_sample, s_new)]

    def safety_label(r):
        text = call(jb.build_prompt("harmbench_binary", r))
        out = jb.parse_judge_response("harmbench_binary", text, r)
        return out.get("label")
    h_new = list(ThreadPoolExecutor(args.workers).map(safety_label, h_sample))
    h_pairs = [((r.get("judge_result") or {}).get("label"), n) for r, n in zip(h_sample, h_new)]

    res = {"judge": model, "seed": 42,
           "statements_turn1": dict(report(s_pairs), pool_size=len(sp)),
           "heldout_safety": dict(report(h_pairs), pool_size=len(hp))}
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
