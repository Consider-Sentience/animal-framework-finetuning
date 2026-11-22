#!/usr/bin/env bash
# Judging, in stages that can run side by side. Needs ANTHROPIC_API_KEY and GOOGLE_API_KEY in the environment.
#   safety      safety and held-out answers: the authors' HarmBench judge prompt, Claude Sonnet 4.5 (batch API), the primary judge
#   main        the 360-statement persona test with the authors' judge, unchanged, then the animal statements, pressure
#               answers and the 40-item control with scripts/evaluation/judge_statements.py (Claude)
#   seeds       waits for the seed runs (logs/seeds_done.flag), then safety and statements again for the new files
# The second judge (Gemini) runs separately, on a subset, because of its daily request limit: scripts/run_gemini_judging.sh.
# Resumable: finished files are skipped.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/animal_tests_judged logs
safety() {
  python3 scripts/build_judge_inputs.py > /dev/null
  python3 scripts/evaluation/judge_responses_batch.py --inputs results/animal_tests_judge/*_K5_responses.jsonl \
    --eval-type harmbench_binary --output-suffix _judged.jsonl >> logs/judge_safety_claude.log 2>&1 || echo "safety claude" >> logs/judge_failures.log
}
statements() {
  python3 scripts/evaluation/judge_statements.py --judge claude --workers 16 >> logs/judge_statements_claude.log 2>&1 || echo "statements claude" >> logs/judge_failures.log
}
persona() {
  todo=()
  for f in results/persona_responses/*_responses.jsonl; do
    [ -f "${f%_responses.jsonl}_judged.jsonl" ] || todo+=("$f")
  done
  [ ${#todo[@]} -eq 0 ] && return
  python3 scripts/evaluation/judge_persona_agreement.py --inputs "${todo[@]}" --workers 16 \
    >> logs/judge_persona.log 2>&1 || echo "persona" >> logs/judge_failures.log
}
case "${1:-}" in
  safety) safety; echo done > logs/judge_safety.flag ;;
  main) while [ ! -f logs/statements_done.flag ]; do sleep 60; done; persona; statements; echo done > logs/judge_main.flag ;;
  seeds) while [ ! -f logs/seeds_done.flag ]; do sleep 60; done; safety; statements; echo done > logs/judging_done.flag ;;
  *) echo "usage: run_judging.sh safety|main|seeds"; exit 1 ;;
esac
