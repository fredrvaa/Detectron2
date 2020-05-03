#!/bin/bash

m='cumulative'
e=1

for d in {1..10}; do
    if [ $d = 1 ]; then
        python train.py --epochs=$e --weights=models/control/O0/best.pth --train=C1 --out_dir=models/$m/CM_$e/CM1 --resume=False
    else
        let prev=$d-1
        python train.py --epochs=$e --weights=models/$m/CM_$e/CM$prev/last.pth --train=C$d --out_dir=models/$m/CM_$e/CM$d --resume=False
    fi
done