#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=20GB
#PBS -l walltime=02:20:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
python3 train.py --dataset cub_200_2011 --model inception_v3.tf_adv_in1k --pretrained --color-jitter --hflip --rotate --train-crop 299 --save-best >> EXB013.log
