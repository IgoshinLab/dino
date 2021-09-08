#!/bin/bash
#SBATCH --partition=commons
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --mail-user=jz79@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=24:00:00
#SBATCH --output=output2
source ~/.bashrc
conda activate dino
srun python main_dino.py --data_path /scratch/jz79/dino/imgnet/ --output_dir /scratch/jz79/dino/ckpts-0907/