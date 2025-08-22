#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=64GB           
#PBS -l walltime=11:00:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training with PKB models
python3 train.py --dataset soyglobal --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --color-jitter --hflip --rotate --save-best >> EXA012.log
