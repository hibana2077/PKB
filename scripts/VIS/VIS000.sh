#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=20GB
#PBS -l walltime=02:30:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

source /scratch/rp06/sl5952/PKB/.venv/bin/activate

cd ../..
python3 vis.py --checkpoint ./outputs/best_resnet50_cotton80.pth --dataset cotton80 --split test --model resnet50 --do-tsne --out-dir ./outputs/vis_cotton_tsne_r50_test