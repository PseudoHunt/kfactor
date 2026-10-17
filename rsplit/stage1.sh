#!/bin/sh
cd /home/kfactor
python rsplit.py ppl NousResearch/Llama-2-7b-hf llama2-7b 2048 40 H,AW64,GW64,R16,R32,R48 rsplit/s1_llama_2048.csv > rsplit/s1_llama_2048.log 2>&1
python rsplit.py ppl Qwen/Qwen1.5-7B qwen1.5-7b 2048 40 H,AW64,GW64,R16,R32,R48 rsplit/s1_qwen_2048.csv > rsplit/s1_qwen_2048.log 2>&1
python rsplit.py ppl Qwen/Qwen1.5-7B qwen1.5-7b 8192 8 H,AW64,GW64,R16,R32,R48 rsplit/s1_qwen_8192.csv > rsplit/s1_qwen_8192.log 2>&1
touch rsplit/stage1.done
