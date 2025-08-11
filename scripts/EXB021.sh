#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=20GB
#PBS -l walltime=00:55:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
python3 train.py --dataset soybean --model densenet161.tv_in1k --color-jitter --hflip --rotate --train-crop 224 --save-best >> EXB021.log