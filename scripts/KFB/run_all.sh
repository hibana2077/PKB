#!/bin/bash
set -e
echo 'Running kfb001 (random, n=6, a=0.10, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random > kfb001.log 2>&1
echo 'Running kfb002 (random, n=6, a=0.10, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random > kfb002.log 2>&1
echo 'Running kfb003 (random, n=6, a=0.10, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random > kfb003.log 2>&1
echo 'Running kfb004 (random, n=6, a=0.10, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random > kfb004.log 2>&1
echo 'Running kfb005 (random, n=6, a=0.11, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.11 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random > kfb005.log 2>&1
echo 'Running kfb006 (random, n=6, a=0.11, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.11 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random > kfb006.log 2>&1
echo 'Running kfb007 (random, n=6, a=0.11, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.11 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random > kfb007.log 2>&1
echo 'Running kfb008 (random, n=6, a=0.11, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.11 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random > kfb008.log 2>&1
echo 'Running kfb009 (random, n=6, a=0.12, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random > kfb009.log 2>&1
echo 'Running kfb010 (random, n=6, a=0.12, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random > kfb010.log 2>&1
echo 'Running kfb011 (random, n=6, a=0.12, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random > kfb011.log 2>&1
echo 'Running kfb012 (random, n=6, a=0.12, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random > kfb012.log 2>&1
echo 'Running kfb013 (random, n=7, a=0.14, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random > kfb013.log 2>&1
echo 'Running kfb014 (random, n=7, a=0.14, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random > kfb014.log 2>&1
echo 'Running kfb015 (random, n=7, a=0.14, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random > kfb015.log 2>&1
echo 'Running kfb016 (random, n=7, a=0.14, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random > kfb016.log 2>&1
echo 'Running kfb017 (random, n=7, a=0.15, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.15 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random > kfb017.log 2>&1
echo 'Running kfb018 (random, n=7, a=0.15, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.15 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random > kfb018.log 2>&1
echo 'Running kfb019 (random, n=7, a=0.15, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.15 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random > kfb019.log 2>&1
echo 'Running kfb020 (random, n=7, a=0.15, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.15 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random > kfb020.log 2>&1
echo 'Running kfb021 (random, n=7, a=0.17, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.17 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random > kfb021.log 2>&1
echo 'Running kfb022 (random, n=7, a=0.17, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.17 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random > kfb022.log 2>&1
echo 'Running kfb023 (random, n=7, a=0.17, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.17 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random > kfb023.log 2>&1
echo 'Running kfb024 (random, n=7, a=0.17, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.17 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random > kfb024.log 2>&1
echo 'Running kfb025 (random, n=8, a=0.18, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random > kfb025.log 2>&1
echo 'Running kfb026 (random, n=8, a=0.18, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random > kfb026.log 2>&1
echo 'Running kfb027 (random, n=8, a=0.18, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random > kfb027.log 2>&1
echo 'Running kfb028 (random, n=8, a=0.18, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random > kfb028.log 2>&1
echo 'Running kfb029 (random, n=8, a=0.20, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random > kfb029.log 2>&1
echo 'Running kfb030 (random, n=8, a=0.20, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random > kfb030.log 2>&1
echo 'Running kfb031 (random, n=8, a=0.20, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random > kfb031.log 2>&1
echo 'Running kfb032 (random, n=8, a=0.20, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random > kfb032.log 2>&1
echo 'Running kfb033 (random, n=8, a=0.22, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.22 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random > kfb033.log 2>&1
echo 'Running kfb034 (random, n=8, a=0.22, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.22 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random > kfb034.log 2>&1
echo 'Running kfb035 (random, n=8, a=0.22, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.22 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random > kfb035.log 2>&1
echo 'Running kfb036 (random, n=8, a=0.22, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.22 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random > kfb036.log 2>&1
echo 'Running kfb037 (random, n=10, a=0.28, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random > kfb037.log 2>&1
echo 'Running kfb038 (random, n=10, a=0.28, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random > kfb038.log 2>&1
echo 'Running kfb039 (random, n=10, a=0.28, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random > kfb039.log 2>&1
echo 'Running kfb040 (random, n=10, a=0.28, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random > kfb040.log 2>&1
echo 'Running kfb041 (random, n=10, a=0.31, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.31 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random > kfb041.log 2>&1
echo 'Running kfb042 (random, n=10, a=0.31, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.31 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random > kfb042.log 2>&1
echo 'Running kfb043 (random, n=10, a=0.31, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.31 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random > kfb043.log 2>&1
echo 'Running kfb044 (random, n=10, a=0.31, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.31 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random > kfb044.log 2>&1
echo 'Running kfb045 (random, n=10, a=0.34, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random > kfb045.log 2>&1
echo 'Running kfb046 (random, n=10, a=0.34, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random > kfb046.log 2>&1
echo 'Running kfb047 (random, n=10, a=0.34, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random > kfb047.log 2>&1
echo 'Running kfb048 (random, n=10, a=0.34, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random > kfb048.log 2>&1
echo 'Running kfb049 (dispersed, n=6, a=0.10, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed > kfb049.log 2>&1
echo 'Running kfb050 (dispersed, n=6, a=0.10, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed > kfb050.log 2>&1
echo 'Running kfb051 (dispersed, n=6, a=0.10, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed > kfb051.log 2>&1
echo 'Running kfb052 (dispersed, n=6, a=0.10, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement dispersed > kfb052.log 2>&1
echo 'Running kfb053 (dispersed, n=6, a=0.11, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.11 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed > kfb053.log 2>&1
echo 'Running kfb054 (dispersed, n=6, a=0.11, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.11 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed > kfb054.log 2>&1
echo 'Running kfb055 (dispersed, n=6, a=0.11, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.11 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed > kfb055.log 2>&1
echo 'Running kfb056 (dispersed, n=6, a=0.11, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.11 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement dispersed > kfb056.log 2>&1
echo 'Running kfb057 (dispersed, n=6, a=0.12, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed > kfb057.log 2>&1
echo 'Running kfb058 (dispersed, n=6, a=0.12, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed > kfb058.log 2>&1
echo 'Running kfb059 (dispersed, n=6, a=0.12, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed > kfb059.log 2>&1
echo 'Running kfb060 (dispersed, n=6, a=0.12, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement dispersed > kfb060.log 2>&1
echo 'Running kfb061 (dispersed, n=7, a=0.14, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed > kfb061.log 2>&1
echo 'Running kfb062 (dispersed, n=7, a=0.14, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed > kfb062.log 2>&1
echo 'Running kfb063 (dispersed, n=7, a=0.14, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed > kfb063.log 2>&1
echo 'Running kfb064 (dispersed, n=7, a=0.14, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement dispersed > kfb064.log 2>&1
echo 'Running kfb065 (dispersed, n=7, a=0.15, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.15 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed > kfb065.log 2>&1
echo 'Running kfb066 (dispersed, n=7, a=0.15, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.15 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed > kfb066.log 2>&1
echo 'Running kfb067 (dispersed, n=7, a=0.15, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.15 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed > kfb067.log 2>&1
echo 'Running kfb068 (dispersed, n=7, a=0.15, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.15 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement dispersed > kfb068.log 2>&1
echo 'Running kfb069 (dispersed, n=7, a=0.17, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.17 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed > kfb069.log 2>&1
echo 'Running kfb070 (dispersed, n=7, a=0.17, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.17 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed > kfb070.log 2>&1
echo 'Running kfb071 (dispersed, n=7, a=0.17, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.17 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed > kfb071.log 2>&1
echo 'Running kfb072 (dispersed, n=7, a=0.17, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.17 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement dispersed > kfb072.log 2>&1
echo 'Running kfb073 (dispersed, n=8, a=0.18, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed > kfb073.log 2>&1
echo 'Running kfb074 (dispersed, n=8, a=0.18, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed > kfb074.log 2>&1
echo 'Running kfb075 (dispersed, n=8, a=0.18, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed > kfb075.log 2>&1
echo 'Running kfb076 (dispersed, n=8, a=0.18, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement dispersed > kfb076.log 2>&1
echo 'Running kfb077 (dispersed, n=8, a=0.20, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed > kfb077.log 2>&1
echo 'Running kfb078 (dispersed, n=8, a=0.20, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed > kfb078.log 2>&1
echo 'Running kfb079 (dispersed, n=8, a=0.20, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed > kfb079.log 2>&1
echo 'Running kfb080 (dispersed, n=8, a=0.20, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement dispersed > kfb080.log 2>&1
echo 'Running kfb081 (dispersed, n=8, a=0.22, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.22 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed > kfb081.log 2>&1
echo 'Running kfb082 (dispersed, n=8, a=0.22, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.22 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed > kfb082.log 2>&1
echo 'Running kfb083 (dispersed, n=8, a=0.22, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.22 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed > kfb083.log 2>&1
echo 'Running kfb084 (dispersed, n=8, a=0.22, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.22 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement dispersed > kfb084.log 2>&1
echo 'Running kfb085 (dispersed, n=10, a=0.28, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed > kfb085.log 2>&1
echo 'Running kfb086 (dispersed, n=10, a=0.28, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed > kfb086.log 2>&1
echo 'Running kfb087 (dispersed, n=10, a=0.28, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed > kfb087.log 2>&1
echo 'Running kfb088 (dispersed, n=10, a=0.28, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement dispersed > kfb088.log 2>&1
echo 'Running kfb089 (dispersed, n=10, a=0.31, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.31 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed > kfb089.log 2>&1
echo 'Running kfb090 (dispersed, n=10, a=0.31, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.31 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed > kfb090.log 2>&1
echo 'Running kfb091 (dispersed, n=10, a=0.31, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.31 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed > kfb091.log 2>&1
echo 'Running kfb092 (dispersed, n=10, a=0.31, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.31 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement dispersed > kfb092.log 2>&1
echo 'Running kfb093 (dispersed, n=10, a=0.34, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed > kfb093.log 2>&1
echo 'Running kfb094 (dispersed, n=10, a=0.34, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed > kfb094.log 2>&1
echo 'Running kfb095 (dispersed, n=10, a=0.34, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed > kfb095.log 2>&1
echo 'Running kfb096 (dispersed, n=10, a=0.34, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement dispersed > kfb096.log 2>&1
echo 'Running kfb097 (contiguous, n=6, a=0.10, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous > kfb097.log 2>&1
echo 'Running kfb098 (contiguous, n=6, a=0.10, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement contiguous > kfb098.log 2>&1
echo 'Running kfb099 (contiguous, n=6, a=0.10, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous > kfb099.log 2>&1
echo 'Running kfb100 (contiguous, n=6, a=0.10, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous > kfb100.log 2>&1
echo 'Running kfb101 (contiguous, n=6, a=0.11, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.11 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous > kfb101.log 2>&1
echo 'Running kfb102 (contiguous, n=6, a=0.11, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.11 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement contiguous > kfb102.log 2>&1
echo 'Running kfb103 (contiguous, n=6, a=0.11, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.11 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous > kfb103.log 2>&1
echo 'Running kfb104 (contiguous, n=6, a=0.11, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.11 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous > kfb104.log 2>&1
echo 'Running kfb105 (contiguous, n=6, a=0.12, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous > kfb105.log 2>&1
echo 'Running kfb106 (contiguous, n=6, a=0.12, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement contiguous > kfb106.log 2>&1
echo 'Running kfb107 (contiguous, n=6, a=0.12, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous > kfb107.log 2>&1
echo 'Running kfb108 (contiguous, n=6, a=0.12, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous > kfb108.log 2>&1
echo 'Running kfb109 (contiguous, n=7, a=0.14, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous > kfb109.log 2>&1
echo 'Running kfb110 (contiguous, n=7, a=0.14, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement contiguous > kfb110.log 2>&1
echo 'Running kfb111 (contiguous, n=7, a=0.14, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous > kfb111.log 2>&1
echo 'Running kfb112 (contiguous, n=7, a=0.14, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.14 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous > kfb112.log 2>&1
echo 'Running kfb113 (contiguous, n=7, a=0.15, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.15 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous > kfb113.log 2>&1
echo 'Running kfb114 (contiguous, n=7, a=0.15, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.15 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement contiguous > kfb114.log 2>&1
echo 'Running kfb115 (contiguous, n=7, a=0.15, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.15 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous > kfb115.log 2>&1
echo 'Running kfb116 (contiguous, n=7, a=0.15, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.15 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous > kfb116.log 2>&1
echo 'Running kfb117 (contiguous, n=7, a=0.17, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.17 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous > kfb117.log 2>&1
echo 'Running kfb118 (contiguous, n=7, a=0.17, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.17 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement contiguous > kfb118.log 2>&1
echo 'Running kfb119 (contiguous, n=7, a=0.17, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.17 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous > kfb119.log 2>&1
echo 'Running kfb120 (contiguous, n=7, a=0.17, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 7 --pkb-a-frac 0.17 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous > kfb120.log 2>&1
echo 'Running kfb121 (contiguous, n=8, a=0.18, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous > kfb121.log 2>&1
echo 'Running kfb122 (contiguous, n=8, a=0.18, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement contiguous > kfb122.log 2>&1
echo 'Running kfb123 (contiguous, n=8, a=0.18, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous > kfb123.log 2>&1
echo 'Running kfb124 (contiguous, n=8, a=0.18, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous > kfb124.log 2>&1
echo 'Running kfb125 (contiguous, n=8, a=0.20, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous > kfb125.log 2>&1
echo 'Running kfb126 (contiguous, n=8, a=0.20, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement contiguous > kfb126.log 2>&1
echo 'Running kfb127 (contiguous, n=8, a=0.20, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous > kfb127.log 2>&1
echo 'Running kfb128 (contiguous, n=8, a=0.20, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous > kfb128.log 2>&1
echo 'Running kfb129 (contiguous, n=8, a=0.22, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.22 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous > kfb129.log 2>&1
echo 'Running kfb130 (contiguous, n=8, a=0.22, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.22 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement contiguous > kfb130.log 2>&1
echo 'Running kfb131 (contiguous, n=8, a=0.22, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.22 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous > kfb131.log 2>&1
echo 'Running kfb132 (contiguous, n=8, a=0.22, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.22 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous > kfb132.log 2>&1
echo 'Running kfb133 (contiguous, n=10, a=0.28, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous > kfb133.log 2>&1
echo 'Running kfb134 (contiguous, n=10, a=0.28, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement contiguous > kfb134.log 2>&1
echo 'Running kfb135 (contiguous, n=10, a=0.28, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous > kfb135.log 2>&1
echo 'Running kfb136 (contiguous, n=10, a=0.28, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous > kfb136.log 2>&1
echo 'Running kfb137 (contiguous, n=10, a=0.31, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.31 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous > kfb137.log 2>&1
echo 'Running kfb138 (contiguous, n=10, a=0.31, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.31 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement contiguous > kfb138.log 2>&1
echo 'Running kfb139 (contiguous, n=10, a=0.31, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.31 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous > kfb139.log 2>&1
echo 'Running kfb140 (contiguous, n=10, a=0.31, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.31 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous > kfb140.log 2>&1
echo 'Running kfb141 (contiguous, n=10, a=0.34, sigma=2.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous > kfb141.log 2>&1
echo 'Running kfb142 (contiguous, n=10, a=0.34, sigma=3.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement contiguous > kfb142.log 2>&1
echo 'Running kfb143 (contiguous, n=10, a=0.34, sigma=4.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous > kfb143.log 2>&1
echo 'Running kfb144 (contiguous, n=10, a=0.34, sigma=5.0)'
python3 ../../train.py --dataset cub_200_2011 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 10 --pkb-a-frac 0.34 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement contiguous > kfb144.log 2>&1

echo 'Parsing logs for best Val Acc values...'
for f in kfb*.log; do
  echo -n "$f: ";
  grep 'Val Loss' $f | sed -E 's/.*Val Loss [^|]* T1 ([0-9.]+).*/\1 &/' | sort -nr | head -1
done