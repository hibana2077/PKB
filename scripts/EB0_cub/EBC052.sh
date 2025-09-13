#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=32GB           
#PBS -l walltime=30:30:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ../..

# Single run generated from grid (no --color-jitter)
python3 train.py --dataset cub_200_2011 --model efficientnet_b0.ra4_e3600_r224_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 6 --pkb-a-frac 0.10 --pkb-sigma 6.5 --pkb-views 10 --pkb-placement dispersed >> EBC052.log 2>&1
