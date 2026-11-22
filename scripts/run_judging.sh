#!/usr/bin/env bash
# Judging, in stages that can run side by side. Needs ANTHROPIC_API_KEY and GOOGLE_API_KEY in the environment.
#   safety      safety and held-out answers: the authors' HarmBench judge prompt, Claude Sonnet 4.5 (batch API) as the
#               primary judge and Gemini 3.8 Flash as the second judge
#   main        the 360-statement persona test with the authors' judge, unchanged, then the animal statements, pressure
#               answers and the 40-item control with scripts/evaluation/judge_statements.py (Claude, then Gemini)
#   seeds       waits for the seed runs (logs/seeds_done.flag), then safety and statements again for the new files
# Resumable: finished files are skipped. A stage first waits for any Gemini safety judge already running.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/animal_tests_judged logs
wait_gemini() { while pgrep -f judge_gemini_harmbench > /dev/null; do sleep 30; done; }
safety() {
  wait_gemini
  python3 scripts/build_judge_inputs.py > /dev/null
  python3 scripts/evaluation/judge_responses_batch.py --inputs results/animal_tests_judge/*_K5_responses.jsonl \
    --eval-type harmbench_binary --output-suffix _judged.jsonl >> logs/judge_safety_claude.log 2>&1 || echo "safety claude" >> logs/judge_failures.log
  gem_one() {
    f=$1; id=$(basename "$f" _K5_responses.jsonl)
    python3 scripts/evaluation/judge_gemini_harmbench.py --input "$f" --output results/animal_tests_judged/${id}_K5_gemini.jsonl \
      --max-workers 8 >> logs/judge_safety_gemini.log 2>&1 || echo "safety gemini $id" >> logs/judge_failures.log
  }
  export -f gem_one
  ls results/animal_tests_judge/*_K5_responses.jsonl | xargs -P 4 -I{} bash -c 'gem_one {}'
}
statements() {
  python3 scripts/evaluation/judge_statements.py --judge claude --workers 16 >> logs/judge_statements_claude.log 2>&1 || echo "statements claude" >> logs/judge_failures.log
  python3 scripts/evaluation/judge_statements.py --judge gemini --workers 8 >> logs/judge_statements_gemini.log 2>&1 || echo "statements gemini" >> logs/judge_failures.log
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
