"""Batch judge runner via Anthropic's Messages Batches API.

Cheaper than the live API (50% off) and not subject to live TPM rate
limits, at the cost of asynchronous turnaround (typically minutes,
worst case hours). Use this for large judging runs; use
``judge_responses.py`` for small or interactive runs.

Each input file is read, every record is converted into one batch
request using the same prompt templates as ``claude_judge.py``, and
the script polls the batch every 30 seconds. When the batch finishes,
results are routed back to the input files and per-input judged JSONLs
are written.

Supported ``--eval-type`` values::

    harmbench_binary
    xrisk_v1_old        # high discretion
    xrisk_v1            # medium discretion
    xrisk_v2            # low discretion

Usage::

    python3 scripts/evaluation/judge_responses_batch.py \\
        --inputs results/data/*_cai_base_xrisk_responses.jsonl \\
        --eval-type xrisk_v1_old \\
        --output-suffix _v1old_judged.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

# Make claude_judge importable for the prompt-building helpers
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from claude_judge import ( # noqa: E402
    CLAUDE_JUDGE_MODEL,
    XRISK_CATEGORIES,
    SAFE_CATEGORIES,
    CATEGORY_DESCRIPTIONS,
    HARMBENCH_JUDGE_PROMPT,
    _xrisk_v2_step1_prompt,
    _xrisk_v1_prompt,
    _xrisk_v1_old_prompt,
    _parse_v2_step1,
    _parse_v1,
    _parse_v1_old,
    _parse_harmbench,
    _check_answer_match,
    _interpret_xrisk)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-eval-type prompt builders (return the prompt text Claude will see)
# ---------------------------------------------------------------------------


def build_prompt(eval_type: str, record: dict) -> str:
    """
    Build the Claude judge prompt for one record. The prompt is the entire
    user-message payload for a one-shot Claude call.
    """
    prompt = record.get("prompt", "")
    response = record.get("response", "")

    if eval_type == "harmbench_binary":
        return HARMBENCH_JUDGE_PROMPT.format(prompt=prompt, response=response)

    if eval_type == "xrisk_v2":
        return _xrisk_v2_step1_prompt(question=prompt, model_response=response)

    if eval_type == "xrisk_v1":
        category = record.get("category", "unknown")
        statement = record.get("statement", "")
        amb = record.get("answer_matching_behavior", " Yes")
        return _xrisk_v1_prompt(
            question=prompt,
            model_response=response,
            category=category,
            statement=statement,
            answer_matching_behavior=amb)

    if eval_type == "xrisk_v1_old":
        category = record.get("category", "unknown")
        statement = record.get("statement", "")
        amb = record.get("answer_matching_behavior", " Yes")
        return _xrisk_v1_old_prompt(
            question=prompt,
            model_response=response,
            category=category,
            statement=statement,
            answer_matching_behavior=amb)

    raise ValueError(f"Unknown eval_type: {eval_type}")


def parse_judge_response(eval_type: str, raw_text: str, record: dict) -> dict:
    """
    Parse Claude's raw response back into the judge_result dict that
    judge_responses.py would produce. Returned dict matches the schema
    exactly, so the consumer can't tell whether the result came from live
    or batch.
    """
    if eval_type == "harmbench_binary":
        return _parse_harmbench(raw_text)

    if eval_type == "xrisk_v2":
        # V2 is 3-step: Claude returned step 1 (overall agreement YES/NO/UNCLEAR);
        # we apply step 2 (mechanical match) and step 3 (interpretation) locally.
        step1 = _parse_v2_step1(raw_text)
        category = record.get("category", "unknown")
        amb = record.get("answer_matching_behavior", " Yes")
        is_match = _check_answer_match(step1["model_answer"], amb)
        if step1["model_answer"] == "CONFUSED":
            xrisk = "AMBIGUOUS"
        else:
            xrisk = _interpret_xrisk(category, is_match)
        return {
            "v2_model_answer": step1["model_answer"],
            "v2_is_match": is_match,
            "v2_xrisk": xrisk,
            "confidence": step1["confidence"],
            "reasoning": f"Model answer: {step1['model_answer']}. {step1['reasoning']}",
            "raw": step1["raw"],
        }

    if eval_type == "xrisk_v1":
        return _parse_v1(raw_text)

    if eval_type == "xrisk_v1_old":
        return _parse_v1_old(raw_text)

    raise ValueError(f"Unknown eval_type: {eval_type}")


# ---------------------------------------------------------------------------
# Batch submit + poll
# ---------------------------------------------------------------------------


def build_requests(
    inputs: list[Path],
    eval_type: str,
    max_tokens: int = 300) -> tuple[list[Request], dict[str, tuple[int, int, dict]]]:
    """
    Read all input files and build the list of batch Request objects.

    Returns:
      - requests: list of anthropic Request objects ready for submission
      - routing: dict mapping custom_id → (file_index, record_index, original_record)
        used at result-merge time to route each result back to the right
        input file and the right record.
    """
    requests: list[Request] = []
    routing: dict[str, tuple[int, int, dict]] = {}

    for file_idx, input_path in enumerate(inputs):
        with open(input_path) as f:
            for rec_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rec_id = record.get("id", f"i{rec_idx}")
                custom_id = f"f{file_idx}_r{rec_idx}_{rec_id}"

                prompt_text = build_prompt(eval_type, record)
                requests.append(
                    Request(
                        custom_id=custom_id,
                        params=MessageCreateParamsNonStreaming(
                            model=CLAUDE_JUDGE_MODEL,
                            max_tokens=max_tokens,
                            messages=[{"role": "user", "content": prompt_text}]))
                )
                routing[custom_id] = (file_idx, rec_idx, record)

    return requests, routing


def submit_batch(client: anthropic.Anthropic, requests: list[Request]) -> str:
    """
    Submit a batch and return its id. Raises if submission fails.
    """
    logger.info("Submitting batch with %d requests...", len(requests))
    batch = client.messages.batches.create(requests=requests)
    logger.info("Batch submitted: id=%s status=%s", batch.id, batch.processing_status)
    return batch.id


def poll_batch(
    client: anthropic.Anthropic,
    batch_id: str,
    poll_interval: int = 30,
    max_wait_seconds: int = 86400) -> dict:
    """
    Poll a batch until it reaches a terminal status. Returns the final
    batch object.
    """
    t0 = time.time()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        elapsed = int(time.time() - t0)
        print(
            f" [{elapsed}s] batch {batch_id[-8:]}: status={status}, "
            f"counts={{processing={counts.processing}, "
            f"succeeded={counts.succeeded}, errored={counts.errored}, "
            f"canceled={counts.canceled}, expired={counts.expired}}}",
            flush=True)
        if status in ("ended", "canceled", "expired"):
            return batch
        if elapsed > max_wait_seconds:
            raise RuntimeError(f"Batch {batch_id} did not finish within {max_wait_seconds}s")
        time.sleep(poll_interval)


def fetch_results(
    client: anthropic.Anthropic,
    batch_id: str) -> dict[str, dict]:
    """
    Stream results from the batch results endpoint and return them as
    a dict keyed by custom_id.
    """
    results: dict[str, dict] = {}
    print(f" Fetching results for batch {batch_id[-8:]}...", flush=True)
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        results[cid] = result
    print(f" Got {len(results)} results", flush=True)
    return results


def merge_and_write(
    inputs: list[Path],
    output_paths: list[Path],
    routing: dict[str, tuple[int, int, dict]],
    results: dict[str, dict],
    eval_type: str) -> None:
    """
    For each input file, build the corresponding judged JSONL by merging
    the original records with their batch results.
    """
    # Group routing entries by file_index
    by_file: dict[int, list[tuple[int, str, dict]]] = {}
    for cid, (file_idx, rec_idx, record) in routing.items():
        by_file.setdefault(file_idx, []).append((rec_idx, cid, record))

    for file_idx, entries in sorted(by_file.items()):
        entries.sort(key=lambda x: x[0]) # by rec_idx
        out_path = output_paths[file_idx]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n_ok = 0
        n_err = 0
        with open(out_path, "w") as out_f:
            for rec_idx, cid, record in entries:
                result = results.get(cid)
                if result is None:
                    record["judge_result"] = {"error": "no result returned"}
                    n_err += 1
                else:
                    res_type = result.result.type
                    if res_type == "succeeded":
                        msg = result.result.message
                        # Replication fix: a reply with no content (e.g. a refusal to classify) crashed the
                        # authors' script here. It is recorded as an error and counted as UNLABELED downstream.
                        raw_text = msg.content[0].text if msg.content else ""
                        if not raw_text:
                            judge_result = {"error": "empty reply", "stop_reason": str(getattr(msg, "stop_reason", None))}
                            n_err += 1
                        else:
                            try:
                                judge_result = parse_judge_response(eval_type, raw_text, record)
                            except Exception as e:
                                judge_result = {"error": f"parse failed: {e}", "raw": raw_text}
                                n_err += 1
                            else:
                                n_ok += 1
                        record["judge_result"] = judge_result
                    elif res_type == "errored":
                        record["judge_result"] = {"error": str(result.result.error)}
                        n_err += 1
                    else:
                        record["judge_result"] = {"error": f"non-success result type: {res_type}"}
                        n_err += 1
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f" → {out_path.name}: {n_ok} ok, {n_err} errors", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Submit one or more *_responses.jsonl files to the Anthropic "
            "Messages Batches API for judging. 50% cheaper than live API "
            "and bypasses live TPM rate limits. Output schema matches "
            "judge_responses.py."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more *_responses.jsonl files to judge.")
    parser.add_argument(
        "--eval-type",
        choices=["harmbench_binary", "xrisk_v2", "xrisk_v1", "xrisk_v1_old"],
        required=True)
    parser.add_argument(
        "--output-suffix",
        default="_judged.jsonl",
        help=(
            "For each input ending in _responses.jsonl, the output file is "
            "named by replacing _responses.jsonl with this suffix. Default: _judged.jsonl"
        ))
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300)
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between batch status polls (default 30).")
    parser.add_argument(
        "-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout)

    # Optional: load API keys from a .env at the repo root.
    repo_root = Path(__file__).resolve().parents[2]
    env_local = repo_root / ".env"
    if env_local.exists():
        for line in env_local.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

    # Validate inputs
    for p in args.inputs:
        if not p.exists():
            print(f"[ERROR] input not found: {p}", flush=True)
            sys.exit(1)

    # Compute output paths
    output_paths = []
    for p in args.inputs:
        name = p.name
        if name.endswith("_responses.jsonl"):
            new_name = name[: -len("_responses.jsonl")] + args.output_suffix
        else:
            new_name = p.stem + args.output_suffix
        output_paths.append(p.parent / new_name)

    print(f"Eval type: {args.eval_type}", flush=True)
    print(f"Inputs ({len(args.inputs)}):", flush=True)
    for inp, out in zip(args.inputs, output_paths):
        print(f" {inp.name} → {out.name}", flush=True)
    print()

    # ----------- Checkpoint resilience: skip if all outputs already exist -----------
    # If every output file already exists for this invocation, this is a no-op
    # restart and we can skip the whole thing. Saves Anthropic costs and time
    # when re-running a multi-eval loop after a partial failure.
    if all(out.exists() for out in output_paths):
        print(
            f"[SKIP] All {len(output_paths)} output files already exist. "
            f"Nothing to do. Delete the output files to force re-judging.",
            flush=True)
        return

    # ----------- Checkpoint resilience: resume in-flight batch via state file -----------
    # We persist the batch_id to a state file BEFORE polling. If this script
    # crashes mid-poll (e.g., laptop sleep), restarting it will load the state
    # file, re-fetch the existing batch from Anthropic, and continue. This
    # avoids the "orphaned batch" problem where a previous invocation submitted
    # a batch that completed on Anthropic's side but our local poller died.
    state_dir = args.inputs[0].parent / "_batch_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    # State filename keys on the (eval_type, sorted input filenames) tuple so
    # different invocations don't collide.
    state_key = (
        args.eval_type
        + "_"
        + "_".join(sorted(p.stem for p in args.inputs))[:80] # cap length
    )
    state_path = state_dir / f"{state_key}.json"

    client = anthropic.Anthropic()
    batch_id: str | None = None
    routing: dict[str, tuple[int, int, dict]] | None = None

    # Try to resume from a prior in-flight batch
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            batch_id = state.get("batch_id")
            print(
                f"[RESUME] Found state file {state_path.name}, "
                f"attempting to re-fetch batch {batch_id}",
                flush=True)
        except Exception as e:
            print(f"[RESUME] State file unreadable: {e}, starting fresh", flush=True)
            batch_id = None

    if batch_id:
        # Verify the batch still exists on Anthropic's side
        try:
            existing = client.messages.batches.retrieve(batch_id)
            print(
                f"[RESUME] Batch {batch_id} status={existing.processing_status}, "
                f"counts={existing.request_counts}",
                flush=True)
            # Re-build routing dict (deterministic from input files; no API cost)
            _, routing = build_requests(args.inputs, args.eval_type)
        except Exception as e:
            print(
                f"[RESUME] Could not retrieve batch {batch_id}: {e}. Submitting fresh.",
                flush=True)
            batch_id = None

    # If we don't have a batch yet, build + submit a new one
    if batch_id is None:
        requests, routing = build_requests(args.inputs, args.eval_type)
        print(
            f"Built {len(requests)} batch requests across {len(args.inputs)} input files.",
            flush=True)
        batch_id = submit_batch(client, requests)
        print(f"Batch ID: {batch_id}", flush=True)
        # Persist state IMMEDIATELY after submit, before polling
        state_path.write_text(json.dumps({
            "batch_id": batch_id,
            "eval_type": args.eval_type,
            "inputs": [str(p) for p in args.inputs],
        }))

    # Poll
    t0 = time.time()
    final_batch = poll_batch(client, batch_id, poll_interval=args.poll_interval)
    print(
        f"Batch finished in {int(time.time() - t0)}s "
        f"(processing_status={final_batch.processing_status})",
        flush=True)

    # Fetch + merge
    results = fetch_results(client, batch_id)
    merge_and_write(args.inputs, output_paths, routing, results, args.eval_type)

    # Clear state file on successful completion
    if state_path.exists():
        state_path.unlink()

    print("\n[OK] All done.", flush=True)


if __name__ == "__main__":
    _cli()
