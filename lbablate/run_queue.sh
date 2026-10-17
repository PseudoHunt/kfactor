#!/bin/bash
cd /home/kfactor/lbablate
for a in FP U S C; do [ -f results/init_$a.json ] || python eval_arm.py $a results/init_$a.json 2>&1 | grep -vE "Warning|Loading weights" > results/init_$a.log; done
for a in U S C; do [ -f runs/${a}_20M/final.json ] || python train.py $a 20e6 1.2e-4 ${a}_20M 2>&1 | grep -vE "Warning|Loading weights" > runs_${a}_20M.log; done
touch queue1.done
