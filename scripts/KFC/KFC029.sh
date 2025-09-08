#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=20GB           
#PBS -l walltime=24:30:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ../..

# Single run generated from grid (no --color-jitter)
python3 train.py --dataset cotton80 --model mobilenetv4_hybrid_medium.ix_e550_r384_in1k --pretrained --augmentation pkb --hflip --rotate --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random >> KFC029.log 2>&1
