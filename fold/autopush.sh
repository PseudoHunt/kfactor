#!/bin/sh
# Commit + push results to GitHub at each phase-2 milestone (code/CSV/PNG/txt only; large arrays are gitignored).
cd /home/kfactor
push() {
  python fold_report.py > fold/fold_ppl_table.txt 2>&1
  python fold_report2.py > fold/fold_ppl_table2.txt 2>&1
  git add -A
  git diff --cached --quiet && return
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "$(date +%T) pushed: $1"
}
for m in "fold/run_all.done|Qwen1.5-7B @8192: old configs A-H complete" \
         "fold/attn_profile_llama2-7b_2048.npz|Attention-mass profile: Llama-2-7b" \
         "fold/sel2_llama2.done|Weighted/constrained selection + werr: Llama-2-7b @2048" \
         "fold/ppl2_llama2_2048.csv|Phase-2 PPL: Llama-2-7b @2048" \
         "fold/attn_profile_qwen1.5-7b_8192.npz|Attention-mass profiles: Qwen1.5-7B @2048 and @8192" \
         "fold/ppl2_qwen_2048.csv|Phase-2 PPL: Qwen1.5-7B @2048" \
         "fold/sel2_qwen8192.done|Weighted/constrained selection + werr: Qwen1.5-7B @8192" \
         "fold/run2.done|Phase-2 PPL: Qwen1.5-7B @8192 (all runs complete)"; do
  f=${m%%|*}; msg=${m#*|}
  until [ -f "$f" ]; do sleep 30; done
  sleep 5
  push "$msg"
done
