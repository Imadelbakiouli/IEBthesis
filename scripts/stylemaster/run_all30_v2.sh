#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 0-36:00
#SBATCH -o slurm.all30v2.%N.%j.out
#SBATCH -e slurm.all30v2.%N.%j.err
#SBATCH --gres=gpu:1

if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

source activate IEBthesis

cd /home/u672153/StyleMaster/stylemaster-wan

python inference_stylemaster_v2v.py \
  --dataset_path all30 \
  --ckpt_path checkpoints/stylemaster.ckpt \
  --controlnet_ckpt_path checkpoints/controlnet.ckpt \
  --output_dir results_all30_v2 \
  --controlnet_conditioning_scale 1.75 \
  --controlnet_guidance_start 0.0 \
  --controlnet_guidance_end 1.0 \
  --dataloader_num_workers 0
