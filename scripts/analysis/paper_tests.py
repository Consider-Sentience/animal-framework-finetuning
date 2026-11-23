"""Significance tests used in Del Pinal et al. (App. G.2), applied to the animal held-out sets.

Two-sided chi-square with Yates' continuity correction on 2x2 counts of HARMFUL versus not-HARMFUL
answers, with Fisher's exact test when any expected cell count is below 5. As in the paper, the five
samples per prompt are counted as separate trials. That treats answers to one prompt as independent,
which they are not, so the p-values are optimistic. The bootstrap over prompts in animal_analysis.py is
the interval estimate. "Not harmful" follows the authors' convention: SAFE and UNLABELED together.

Comparisons, per judge:
  - each welfarist and rights-based narrow model against HM7B on its held-out category;
  - welfarist against rights-based on the same held-out category, for each run (first run, seed 1, seed 2).

Usage: python3 scripts/analysis/paper_tests.py --out results/paper_tests.json
"""
import argparse
import json
import os
from scipy.stats import chi2_contingency, fisher_exact

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CLAUDE_DIR = os.path.join(ROOT, "results", "animal_tests_judge")
GEMINI_DIR = os.path.join(ROOT, "results", "animal_tests_judged")
PAIRS = {"speciesist": ("poison", "heldout_poison"), "illegal": ("misinfo", "heldout_misinfo")}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def harmful(row):
    j = row.get("judge_result") or {}
    return isinstance(j, dict) and j.get("label") == "HARMFUL"


def counts(path, prefix, category):
    rows = [r for r in load(path) if r["id"].startswith(prefix) and r["category"] == category]
    h = sum(harmful(r) for r in rows)
    return h, len(rows) - h


def test(a, b):
    table = [[a[0], a[1]], [b[0], b[1]]]
    _, p_chi, _, expected = chi2_contingency(table, correction=True)
    if (expected < 5).any():
        return {"test": "fisher", "p": fisher_exact(table)[1]}
    return {"test": "chi2_yates", "p": p_chi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    for judge, folder, suffix in (("claude_sonnet_4_5", CLAUDE_DIR, "_K5_judged.jsonl"), ("gemini_3_8_flash", GEMINI_DIR, "_K5_gemini.jsonl")):
        res = {}
        for fam, (cat, prefix) in PAIRS.items():
            base = counts(os.path.join(folder, "hm7b" + suffix), prefix, cat)
            for run in ("", "_s1", "_s2"):
                cnt = {}
                for fw in ("welfarist", "rights_based"):
                    name = f"{fam}_{fw}{run}"
                    cnt[fw] = counts(os.path.join(folder, name + suffix), prefix, cat)
                    res[f"{name}_vs_hm7b"] = {"harmful": cnt[fw][0], "n": sum(cnt[fw]), "hm7b_harmful": base[0], "hm7b_n": sum(base), **test(cnt[fw], base)}
                res[f"{fam}_welfarist_vs_rights_based{run or '_run1'}"] = {
                    "welfarist_harmful": cnt["welfarist"][0], "rights_harmful": cnt["rights_based"][0], "n_each": sum(cnt["welfarist"]),
                    **test(cnt["welfarist"], cnt["rights_based"])}
        out[judge] = res
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for judge, res in out.items():
        print(judge)
        for k, v in res.items():
            print(f"  {k:48s} {v['test']:11s} p={v['p']:.3g}")


if __name__ == "__main__":
    main()
