#!/bin/bash
cd /home/kfactor/lbablate
for a in U S; do [ -f runs/${a}_50M/final.json ] || python train.py $a 50e6 1.2e-4 ${a}_50M 2>&1 | grep -vE "Warning|Loading weights" > runs_${a}_50M.log; done
touch queue2.done
