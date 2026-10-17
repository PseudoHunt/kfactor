#!/bin/sh
cd /home/kfactor
while [ ! -f rsplit/stage2.done ]; do sleep 10; done
python rsplit.py probe NousResearch/Llama-2-7b-hf llama2-7b 2048 8 R16,R32,R48,R16L300,R16L600,R32L300,R32L600,R48L300,R48L600,R16L300N,R16L600N,R32L300N,R32L600N,R48L300N,R48L600N,AW64,GW64 rsplit/p_llama_2048 > rsplit/p_llama_2048.log 2>&1
python rsplit.py probe Qwen/Qwen1.5-7B qwen1.5-7b 2048 8 R16,R32,R48,R16L300,R16L600,R32L300,R32L600,R48L300,R48L600,R16L300N,R16L600N,R32L300N,R32L600N,R48L300N,R48L600N,AW64,GW64 rsplit/p_qwen_2048 > rsplit/p_qwen_2048.log 2>&1
python rsplit.py probe Qwen/Qwen1.5-7B qwen1.5-7b 8192 2 R16,R32,R48,R16L300,R16L600,R32L300,R32L600,R48L300,R48L600,R16L300N,R16L600N,R32L300N,R32L600N,R48L300N,R48L600N,AW64,GW64 rsplit/p_qwen_8192 > rsplit/p_qwen_8192.log 2>&1
touch rsplit/stage3.done
