#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=64GB
#PBS -l walltime=18:22:22
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ..
# Run training with PKB models (SoyGene - TVE003)
python3 train.py --dataset soygene --model tiny_vit_21m_384.dist_in22k_ft_in1k --pretrained --augmentation pkb --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 4 --pkb-placement dispersed --hflip --rotate --save-best >> TVE003.log
