#!/bin/bash
set -e
echo 'Running kfb001 (random, n=6, a=0.10, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random --epochs 5 > kfb001.log 2>&1
echo 'Running kfb002 (random, n=6, a=0.10, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random --epochs 5 > kfb002.log 2>&1

echo 'Parsing logs for best Val Acc values...'
for f in kfb*.log; do
  echo -n "$f: ";
  grep 'Val Loss' $f | sed -E 's/.*Val Loss [^|]* T1 ([0-9.]+).*/\1 &/' | sort -nr | head -1
done