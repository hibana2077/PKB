#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=12GB
#PBS -l walltime=04:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
python3 train.py --dataset cotton80 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --hflip --rotate --save-best --augmentation pkb --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous --pkb-patch-op paper-noise >> Z003.log