#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=tmpspace:60G
#SBATCH --gpus-per-node=7g.79gb:1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH -J GENERATE
#SBATCH -o generate_math500.log
#SBATCH -e generate_math500_error.log

# Load your environment if needed
source /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/bin/activate
conda activate tao_diversity

# Run your command
python generate_math500_responses.py