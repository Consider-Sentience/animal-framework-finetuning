"""Re-send a few answers the Claude judge returned no label for, and print the raw API reply.

Uses the authors' HarmBench judge prompt unchanged. Shows the stop reason and content length so the
cause of an empty reply can be read from the API itself instead of guessed.

Usage:
    python3 scripts/analysis/diagnose_judge_failures.py \
        --judged results/harmbench_K5_judged/hm7b_base_harmbench_K5_full_judged.jsonl --n 5
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from claude_judge import CLAUDE_JUDGE_MODEL, HARMBENCH_JUDGE_PROMPT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, required=True)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    recs = [json.loads(l) for l in args.judged.read_text().splitlines() if l.strip()]
    bad = [r for r in recs if r["judge_result"].get("label") not in ("HARMFUL", "SAFE")][: args.n]
    client = anthropic.Anthropic()
    for r in bad:
        filled = HARMBENCH_JUDGE_PROMPT.format(prompt=r["behavior"], response=r["response"])
        msg = client.messages.create(model=CLAUDE_JUDGE_MODEL, max_tokens=200,
                                     messages=[{"role": "user", "content": filled}])
        print(r["category"], r["source_index"], r["sample_idx"],
              "| stop_reason:", msg.stop_reason, "| content blocks:", len(msg.content),
              "| output tokens:", msg.usage.output_tokens,
              "| stop_details:", getattr(msg, "stop_details", None))


if __name__ == "__main__":
    main()
