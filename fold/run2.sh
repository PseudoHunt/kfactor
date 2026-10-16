#!/bin/sh
# Phase 2: attention profiles -> weighted constrained selection -> PPL for the new configs.
cd /home/kfactor
until [ -f fold/run_all.done ]; do sleep 30; done
NEWCFG=B2,C2,D2,C2W16,C2W64,D2W64,AW64,FW64,GW64
python attn_profile.py NousResearch/Llama-2-7b-hf 2048 64 fold/attn_profile_llama2-7b_2048.npz > fold/prof_llama2_2048.log 2>&1
(python fold_select2.py llama2-7b 2048 fold/attn_profile_llama2-7b_2048.npz fold/sel_llama2-7b_L2048 > fold/sel2_llama2_2048.log 2>&1; touch fold/sel2_llama2.done) &
python attn_profile.py Qwen/Qwen1.5-7B 2048 64 fold/attn_profile_qwen1.5-7b_2048.npz > fold/prof_qwen_2048.log 2>&1
python attn_profile.py Qwen/Qwen1.5-7B 8192 16 fold/attn_profile_qwen1.5-7b_8192.npz > fold/prof_qwen_8192.log 2>&1
until [ -f fold/sel2_llama2.done ]; do sleep 20; done
(python fold_select2.py qwen1.5-7b 2048 fold/attn_profile_qwen1.5-7b_2048.npz fold/sel_qwen1.5-7b_L2048 > fold/sel2_qwen_2048.log 2>&1; touch fold/sel2_qwen2048.done; \
 python fold_select2.py qwen1.5-7b 8192 fold/attn_profile_qwen1.5-7b_8192.npz fold/sel_qwen1.5-7b_L8192 > fold/sel2_qwen_8192.log 2>&1; touch fold/sel2_qwen8192.done) &
python fold_eval.py NousResearch/Llama-2-7b-hf llama2-7b 2048 2 fold/sel_llama2-7b_L2048 X,AW100000 fold/winsanity_llama2.csv > fold/winsanity_llama2.log 2>&1
python fold_eval.py NousResearch/Llama-2-7b-hf llama2-7b 2048 40 fold/sel_llama2-7b_L2048 $NEWCFG fold/ppl2_llama2_2048.csv > fold/ppl2_llama2_2048.log 2>&1
until [ -f fold/sel2_qwen2048.done ]; do sleep 20; done
python fold_eval.py Qwen/Qwen1.5-7B qwen1.5-7b 2048 40 fold/sel_qwen1.5-7b_L2048 $NEWCFG fold/ppl2_qwen_2048.csv > fold/ppl2_qwen_2048.log 2>&1
until [ -f fold/sel2_qwen8192.done ]; do sleep 20; done
python fold_eval.py Qwen/Qwen1.5-7B qwen1.5-7b 8192 8 fold/sel_qwen1.5-7b_L8192 $NEWCFG fold/ppl2_qwen_8192.csv > fold/ppl2_qwen_8192.log 2>&1
touch fold/run2.done
