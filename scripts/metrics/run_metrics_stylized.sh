#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 0-12:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --gres=gpu:1

if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

source activate IEBthesis

cd ~/tapnet
python3 run_metrics_stylized.py

