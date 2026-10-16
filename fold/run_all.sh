#!/bin/sh
cd /home/kfactor
python fold_eval.py NousResearch/Llama-2-7b-hf llama2-7b 2048 40 fold/sel_llama2-7b_L2048 H,X,A,F,G,B,C,D,E fold/ppl_llama2_2048.csv > fold/ppl_llama2_2048.log 2>&1
until grep -q "^wrote" fold/sel_qwen_2048.log; do sleep 20; done
python fold_eval.py Qwen/Qwen1.5-7B qwen1.5-7b 2048 40 fold/sel_qwen1.5-7b_L2048 H,X,A,F,G,B,C,D,E fold/ppl_qwen_2048.csv > fold/ppl_qwen_2048.log 2>&1
until grep -q "^wrote" fold/sel_qwen_8192.log 2>/dev/null; do sleep 20; done
python fold_eval.py Qwen/Qwen1.5-7B qwen1.5-7b 8192 8 fold/sel_qwen1.5-7b_L8192 H,X,A,F,G,B,C,D,E fold/ppl_qwen_8192.csv > fold/ppl_qwen_8192.log 2>&1
echo ALLDONE > fold/run_all.done
