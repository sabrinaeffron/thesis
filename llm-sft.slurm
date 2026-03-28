#!/bin/bash
#SBATCH --job-name=ood_q5    # create a short name for your job
#SBATCH --account=griffith 
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:a100:1        # number of gpus per node
#SBATCH --time=00:45:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=se1854@princeton.edu

module purge
module load anaconda3/2024.6
conda activate sft-env

python test_ood_logits_checkpoints2.py
