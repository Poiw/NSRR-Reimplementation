#!/bin/sh
#SBATCH -c 14
#SBATCH --gres=gpu:1 -p g24

python train.py -c config.json