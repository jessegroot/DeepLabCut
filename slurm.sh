#!/bin/sh
# SBATCH -J demo_openfield
# SBATCH -o demo_openfield.out
# SBATCH -e demo_openfield.err
# SBATCH -N 1
# SBTACH --mem=16000
# SBATCH --partition=gpu

source /data/conda/bin/activate dlc
python /home/11012579/training_network.py
