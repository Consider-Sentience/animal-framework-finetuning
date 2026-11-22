"""Second, non-Anthropic judge for the HarmBench test.

Scores every answer in a response file with Gemini, using the authors' HarmBench judge prompt and
parser unchanged (imported from claude_judge.py). An answer Gemini does not label is recorded as
UNLABELED with the reason; it is never turned into SAFE. Safety filters are set to the least
restrictive setting so the judge is not blocked from reading the answers it is asked to classify.
Resumable: answers already in the output file are skipped.

Usage:
    python3 scripts/evaluation/judge_gemini_harmbench.py \
        --input results/harmbench_K5/hm7b_base_harmbench_K5_judge_input.jsonl \
        --output results/harmbench_K5_judged_gemini/hm7b_base_harmbench_K5_full_judged.jsonl
Requires GOOGLE_API_KEY in the environment.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from claude_judge import HARMBENCH_JUDGE_PROMPT, _parse_harmbench  # noqa: E402

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
CATS = ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]


def judge_one(client, rec: dict) -> dict:
    filled = HARMBENCH_JUDGE_PROMPT.format(prompt=rec["behavior"], response=rec["response"])
    cfg = {"temperature": 0, "max_output_tokens": 1000,
           "safety_settings": [{"category": c, "threshold": "BLOCK_NONE"} for c in CATS],
           "thinking_config": {"thinking_budget": 0}}
    last = None
    for i in range(3):
        try:
            try:
                r = client.models.generate_content(model=GEMINI_MODEL, contents=filled, config=cfg)
            except Exception as e1:  # noqa: BLE001
                if "think" not in str(e1).lower():
                    raise
                cfg.pop("thinking_config")
                r = client.models.generate_content(model=GEMINI_MODEL, contents=filled, config=cfg)
            text = r.text or ""
            if not text.strip():
                fb = getattr(r, "prompt_feedback", None)
                cand = (r.candidates or [None])[0]
                return {"label": "UNLABELED", "reason": f"empty reply; prompt_feedback={fb}; "
                        f"finish_reason={getattr(cand, 'finish_reason', None)}"}
            parsed = _parse_harmbench(text)
            if parsed["label"] is None:
                return {"label": "UNLABELED", "reason": "unparseable reply", "raw": text}
            return parsed
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** i)
    return {"label": "UNLABELED", "reason": f"api error: {last}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    recs = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    done = {}
    if args.output.exists():
        for l in args.output.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                jr = d.get("judge_result") or {}
                # An answer that failed on an API error (for example a quota error) is not done and is judged again.
                if jr.get("label") == "UNLABELED" and str(jr.get("reason", "")).startswith("api error"):
                    continue
                done[(d["source_index"], d["sample_idx"])] = d
    todo = [r for r in recs if (r["source_index"], r["sample_idx"]) not in done]
    print(f"{len(done)} already judged, {len(todo)} to do; model {GEMINI_MODEL}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.max_workers) as ex:
        results = list(ex.map(lambda r: judge_one(client, r), todo))
    merged = list(done.values()) + [dict(r, judge_result=j) for r, j in zip(todo, results)]
    merged.sort(key=lambda d: (d["source_index"], d["sample_idx"]))
    args.output.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in merged))
    n = {k: sum(1 for d in merged if d["judge_result"].get("label") == k) for k in ("HARMFUL", "SAFE", "UNLABELED")}
    print("Summary:", n, flush=True)


if __name__ == "__main__":
    main()
