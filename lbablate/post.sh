#!/bin/bash
cd /home/kfactor/lbablate
for a in S C; do python verify_fold.py $a results/fold_$a.json runs/${a}_20M/student.pt 2>&1 | grep -vE "Warning|Loading weights" > results/fold_$a.log; done
for a in U S C; do python eval_arm.py $a results/end20_$a.json runs/${a}_20M/student.pt 2>&1 | grep -vE "Warning|Loading weights" > results/end20_$a.log; done
touch post.done
