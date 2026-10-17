#!/bin/bash
cd /home/kfactor/lbablate
while [ ! -f runs/U_20M_seed1/final.json ]; do sleep 20; done
: > results/localg_p0.csv; : > results/localg_p1.csv
L0=$(python -c "print(','.join(str(l) for l in range(2,24,2)))"); L1=$(python -c "print(','.join(str(l) for l in range(3,24,2)))")
python localqat_g.py $L0 results/localg_p0.csv S2,S4,S8 > results/localg_p0.log 2>&1 &
python localqat_g.py $L1 results/localg_p1.csv S2,S4,S8 > results/localg_p1.log 2>&1 &
wait
python local_attn_g.py U,S,S2,S4,S8 results/local_attn_g.csv > results/local_attn_g.log 2>&1
for g in 1 2 4 8; do python fold_g.py $g results/fold_g$g.json > results/fold_g$g.log 2>&1; done
touch run_g.done
