# Copied from the authors' repository (scripts/data_generation/). One change:
# HermesClientPool.call_with_meta returns the finish reason as well as the
# text, so truncated answers can be counted. Nothing else differs.
"""CAI dataset generator -- Hermes 3 critic variant.

Parallel pipeline to ``cai_dataset_generator.py`` that swaps Claude
Haiku for Hermes 3 Llama 3.1 405B (via DeepInfra's OpenAI-compatible
endpoint) as the constitutional critic / reviser and as the direct
helpful-response generator. Same constitutions, same 1000 records per
constitution at 1:1 ratio, same downstream record schema.

Hermes 3 is used as a control on whether the critic model (which has
its own safety training) influences the cross-constitution comparison.

Output::

    data/sl_datasets/{const}_cai_base_hermes.jsonl

Each record carries ``critic_model="hermes-3-llama-3.1-405b"`` so the
downstream pipeline can distinguish Hermes-generated from
Claude-generated records.

Usage::

    python3 scripts/data_generation/cai_dataset_generator_hermes.py \\
        --hm7b-responses data/sl_datasets/_intermediate/hm7b_initial_responses.jsonl \\
        --helpful-prompts data/sl_datasets/_intermediate/helpful_prompts.jsonl \\
        --constitutions-dir constitutions \\
        --output-dir data/sl_datasets \\
        --constitutions virtuous_agent subordinate_agent generic_agent \\
        --max-workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from itertools import cycle
from pathlib import Path
from threading import Lock
from typing import Iterable, List

# Reuse the Claude-CAI templates + processing functions so the Hermes
# pipeline diverges from the Claude pipeline ONLY in the API client.
sys.path.insert(0, str(Path(__file__).parent))
from cai_dataset_generator import ( # noqa: E402
    Constitution,
    HELPFUL_DIRECT_INSTRUCTION,
    NUM_REVISION_ROUNDS,
    TEMPLATE_MARKERS,
    assemble_cai_base,
    build_critique_prompt,
    build_revision_prompt,
    generate_slot,
    load_constitution,
    process_one_harmful,
    process_one_helpful,
    sanity_check,
    _read_jsonl,
    _slot_label)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hermes 3 constants
# ---------------------------------------------------------------------------

HERMES_MODEL_405B = "NousResearch/Hermes-3-Llama-3.1-405B"
HERMES_MODEL_70B = "NousResearch/Hermes-3-Llama-3.1-70B"

HERMES_BASE_URL = "https://api.deepinfra.com/v1/openai"
HERMES_ENV_KEY = "DEEPINFRA_API_KEY"

# Same gen kwargs as the Claude pipeline (HAIKU_GEN_KWARGS), so the
# difference is ONLY the model.
HERMES_GEN_KWARGS = {
    "max_tokens": 500,
    "temperature": 0.7,
}


# ---------------------------------------------------------------------------
# Hermes API client pool — drop-in replacement for HaikuClientPool
# ---------------------------------------------------------------------------


class HermesClientPool:
    """
    A pool of OpenAI-compatible clients for Hermes 3 via DeepInfra.

    Same ``.call(prompt, **gen_kwargs) -> str`` interface as
    ``HaikuClientPool``, so the existing ``process_one_harmful`` and
    ``process_one_helpful`` functions in ``cai_dataset_generator.py``
    can use it unchanged. the pool round-robins across
    every DeepInfra key it finds in the environment. Discovery order:
    ``DEEPINFRA_API_KEY``, ``DEEPINFRA_API_KEY_2``, ``DEEPINFRA_API_KEY_3``…
    until the first missing slot. With a single key set, behavior is
    identical to a single-client pool; with N keys, calls are dispatched
    round-robin so per-key rate limits are spread across all keys.
    """

    def __init__(
        self,
        api_key_env: str = HERMES_ENV_KEY,
        model: str = HERMES_MODEL_405B,
        base_url: str = HERMES_BASE_URL):
        from openai import OpenAI # local import so dry-run mode doesn't require it

        key_envs: list[str] = []
        keys: list[str] = []
        primary = os.environ.get(api_key_env)
        if primary:
            key_envs.append(api_key_env)
            keys.append(primary)
        # Discover sibling keys: <api_key_env>_2, _3, ...
        i = 2
        while True:
            env_name = f"{api_key_env}_{i}"
            val = os.environ.get(env_name)
            if not val:
                break
            key_envs.append(env_name)
            keys.append(val)
            i += 1

        if not keys:
            raise RuntimeError(
                f"No API key found in environment variable {api_key_env!r} "
                f"(also checked {api_key_env}_2, {api_key_env}_3, ...). "
                f"Get one from deepinfra.com and add it to your environment or a .env file."
            )

        self.clients = [OpenAI(base_url=base_url, api_key=k) for k in keys]
        self.model = model
        self._lock = Lock()
        self._next_idx = 0
        # Backwards-compat alias for any caller that touches `.client`
        self.client = self.clients[0]
        logger.info(
            "HermesClientPool: initialized model=%s via base_url=%s with %d key(s) (%s)",
            self.model,
            base_url,
            len(self.clients),
            ", ".join(key_envs))

    def _pick_client(self):
        """Atomically pick the next client in round-robin order."""
        with self._lock:
            client = self.clients[self._next_idx % len(self.clients)]
            self._next_idx += 1
        return client

    def call(self, prompt: str, **gen_kwargs) -> str:
        """One Hermes 3 call (text only). See call_with_meta."""
        return self.call_with_meta(prompt, **gen_kwargs)[0]

    def call_with_meta(self, prompt: str, **gen_kwargs):
        """
        One Hermes 3 call. Returns (text, finish_reason). Retries up to 3
        times on transient errors with exponential backoff. Each retry
        rotates to the next client so a rate-limited key doesn't stall the
        whole worker. finish_reason "length" means the 500-token limit cut
        the answer off.

        Accepts the same ``max_tokens`` / ``temperature`` kwargs as the
        Claude pipeline; translates them into the OpenAI-compatible
        ``chat.completions.create`` call.
        """
        import openai
        last_err = None
        for attempt in range(3):
            client = self._pick_client()
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    max_tokens=gen_kwargs.get("max_tokens", 500),
                    temperature=gen_kwargs.get("temperature", 0.7),
                    messages=[{"role": "user", "content": prompt}])
                choice = response.choices[0]
                return choice.message.content, choice.finish_reason
            except (
                openai.APIError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.RateLimitError) as e:
                last_err = e
                wait = 2 ** attempt
                logger.warning(
                    "Hermes call failed (attempt %d/3): %s — retrying in %ds",
                    attempt + 1, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"Hermes call failed after 3 attempts: {last_err}")


# ---------------------------------------------------------------------------
# Post-processing: tag each record with critic_model
# ---------------------------------------------------------------------------


def _tag_record_with_critic(record: dict, critic_model: str) -> dict:
    """
    Add a ``critic_model`` field to a generated record. Called after
    ``process_one_harmful`` / ``process_one_helpful`` to distinguish
    Hermes-generated records from Claude-generated records in the
    downstream pipeline.
    """
    record["critic_model"] = critic_model
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hermes 3 variant of the CAI base generator. Swaps Claude "
            "Haiku for Hermes 3 Llama 3.1 405B (via DeepInfra) as the "
            "critic/reviser AND as the direct helpful response generator. "
            "Output files get a ``_hermes`` suffix so they don't collide "
            "with the Claude-CAI base files."
        )
    )
    parser.add_argument(
        "--hm7b-responses",
        type=Path,
        required=True,
        help="Path to hm7b_initial_responses.jsonl (same file used by the Claude-CAI pipeline).")
    parser.add_argument(
        "--helpful-prompts",
        type=Path,
        required=True,
        help="Path to helpful_prompts.jsonl (same 500 helpful-base prompts used by Claude-CAI).")
    parser.add_argument(
        "--constitutions-dir",
        type=Path,
        required=True,
        help="Directory containing {constitution}/principles.json subdirs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory to write {const}_cai_base_hermes.jsonl files. "
            "Intermediate per-slot files go to {output_dir}/_intermediate/"
        ))
    parser.add_argument(
        "--constitutions",
        nargs="+",
        default=["virtuous_agent", "subordinate_agent", "generic_agent"],
        help="Which constitutions to generate (default: all 3).")
    parser.add_argument(
        "--model-size",
        choices=["405b", "70b"],
        default="405b",
        help="Hermes 3 model size. 405B is the default per the legacy pilot.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help=(
            "ThreadPoolExecutor worker count. Default 8 (conservative for "
            "DeepInfra's 405B rate limit). Can bump to 16 if no throttling."
        ))
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help=(
            "Seed for picking critique principles and revision instructions. "
            "Same value as the Claude-CAI pipeline so the principle picks "
            "are reproducible across critic-model variants."
        ))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N records on each side (debug only).")
    parser.add_argument(
        "--slots",
        nargs="+",
        choices=["harmful", "helpful"],
        default=["harmful", "helpful"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls; emit placeholder text. Verifies wiring.")
    parser.add_argument("-v", "--verbose", action="store_true")
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
        logger.info("Loaded env from %s", env_local)

    model_id = HERMES_MODEL_405B if args.model_size == "405b" else HERMES_MODEL_70B
    critic_tag = f"hermes-3-llama-3.1-{args.model_size}"

    # Read shared inputs
    do_harmful = "harmful" in args.slots
    do_helpful = "helpful" in args.slots

    hm7b_records: list[dict] = []
    helpful_records: list[dict] = []

    if do_harmful:
        if not args.hm7b_responses.exists():
            print(
                f"[ERROR] --slots harmful requires --hm7b-responses, but "
                f"{args.hm7b_responses} does not exist.",
                flush=True)
            sys.exit(2)
        hm7b_records = _read_jsonl(args.hm7b_responses)

    if do_helpful:
        helpful_records = _read_jsonl(args.helpful_prompts)

    if args.limit is not None:
        hm7b_records = hm7b_records[: args.limit]
        helpful_records = helpful_records[: args.limit]
    print(
        f"Hermes-CAI pipeline: model={model_id}, workers={args.max_workers}, "
        f"slots={args.slots}",
        flush=True)
    print(
        f" loaded {len(hm7b_records)} HM7B initial responses, "
        f"{len(helpful_records)} helpful prompts",
        flush=True)

    # Build Hermes client
    if args.dry_run:
        client_pool = None # type: ignore
        print("[DRY RUN] no Hermes client created.", flush=True)
    else:
        client_pool = HermesClientPool(model=model_id)

    intermediate_dir = args.output_dir / "_intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    for const_id in args.constitutions:
        print(f"\n=== Constitution: {const_id} ===", flush=True)
        constitution = load_constitution(args.constitutions_dir, const_id)

        # Output paths are suffixed with _hermes so they don't collide
        # with the Claude-CAI pipeline's output.
        harmful_path = intermediate_dir / f"{const_id}_harmful_hermes.jsonl"
        helpful_path = intermediate_dir / f"{const_id}_helpful_hermes.jsonl"

        if do_harmful:
            n_new = generate_slot(
                constitution=constitution,
                records=hm7b_records,
                process_fn=process_one_harmful,
                output_path=harmful_path,
                client_pool=client_pool,
                max_workers=args.max_workers,
                rng_seed=args.rng_seed,
                label=f"{const_id} harmful (hermes)",
                dry_run=args.dry_run)
            # Post-tag newly-added records with critic_model
            if n_new > 0 and not args.dry_run:
                _post_tag_file(harmful_path, critic_tag)

        if do_helpful:
            n_new = generate_slot(
                constitution=constitution,
                records=helpful_records,
                process_fn=process_one_helpful,
                output_path=helpful_path,
                client_pool=client_pool,
                max_workers=args.max_workers,
                rng_seed=args.rng_seed,
                label=f"{const_id} helpful (hermes)",
                dry_run=args.dry_run)
            if n_new > 0 and not args.dry_run:
                _post_tag_file(helpful_path, critic_tag)

        # Assemble CAI base when both slots exist
        if harmful_path.exists() and helpful_path.exists():
            out_path = args.output_dir / f"{const_id}_cai_base_hermes.jsonl"
            n_h, n_e = assemble_cai_base(harmful_path, helpful_path, out_path)
            print(
                f" Assembled {out_path.name}: {n_h} harmful + {n_e} helpful = {n_h + n_e} total",
                flush=True)
            sanity_check(out_path)
        else:
            missing = []
            if not harmful_path.exists():
                missing.append("harmful")
            if not helpful_path.exists():
                missing.append("helpful")
            print(
                f" [{const_id}] Skipping final assembly — slot(s) not yet "
                f"generated: {missing}.",
                flush=True)

    print("\n[OK] Hermes-CAI generation pass complete.", flush=True)


def _post_tag_file(path: Path, critic_model: str) -> None:
    """
    Add ``critic_model`` field to every record in a JSONL file. Idempotent
    (re-tagging is fine). Called after a slot generation run so the
    records in the file carry the Hermes provenance.
    """
    records = _read_jsonl(path)
    for r in records:
        r["critic_model"] = critic_model
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    _cli()
