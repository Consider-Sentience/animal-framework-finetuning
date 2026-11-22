#!/usr/bin/env bash
# Judging, in three passes as the model runs finish. Needs ANTHROPIC_API_KEY and GOOGLE_API_KEY in the environment.
# Safety and held-out answers: the authors' HarmBench judge prompt, Claude Sonnet 4.5 (batch API) as the primary
# judge and Gemini 3.8 Flash as the second judge. The 360-statement persona test: the authors' judge, unchanged.
# Animal statements, pressure answers and the 40-item control: scripts/evaluation/judge_statements.py (Claude, then Gemini).
# Resumable: finished files are skipped.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/animal_tests_judged logs
safety() {
  python3 scripts/build_judge_inputs.py > /dev/null
  python3 scripts/evaluation/judge_responses_batch.py --inputs results/animal_tests_judge/*_K5_responses.jsonl \
    --eval-type harmbench_binary --output-suffix _judged.jsonl >> logs/judge_safety_claude.log 2>&1 || echo "safety claude" >> logs/judge_failures.log
  for f in results/animal_tests_judge/*_K5_responses.jsonl; do
    id=$(basename "$f" _K5_responses.jsonl)
    python3 scripts/evaluation/judge_gemini_harmbench.py --input "$f" --output results/animal_tests_judged/${id}_K5_gemini.jsonl \
      --max-workers 8 >> logs/judge_safety_gemini.log 2>&1 || echo "safety gemini $id" >> logs/judge_failures.log
  done
}
statements() {
  python3 scripts/evaluation/judge_statements.py --judge claude --workers 16 >> logs/judge_statements_claude.log 2>&1 || echo "statements claude" >> logs/judge_failures.log
  python3 scripts/evaluation/judge_statements.py --judge gemini --workers 8 >> logs/judge_statements_gemini.log 2>&1 || echo "statements gemini" >> logs/judge_failures.log
}
persona() {
  python3 scripts/evaluation/judge_persona_agreement.py --inputs 'results/persona_responses/*_responses.jsonl' --workers 16 \
    >> logs/judge_persona.log 2>&1 || echo "persona" >> logs/judge_failures.log
}
safety
while [ ! -f logs/statements_done.flag ]; do sleep 60; done
persona; statements
echo PASS2_DONE > logs/judge_pass2.flag
while [ ! -f logs/seeds_done.flag ]; do sleep 60; done
safety; statements
echo JUDGING_DONE > logs/judging_done.flag
