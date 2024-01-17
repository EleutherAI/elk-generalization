#!/bin/bash
#SBATCH --job-name=lora-sft
#SBATCH --partition=a40x
#SBATCH --cpus-per-task=12          # Number of cores per tasks
#SBATCH --gres=gpu:1                 # Number of gpus
#SBATCH --output=/admin/home-alexmallen/sft-logs/%j.out      # Set this dir where you want slurm outs to go
#SBATCH --error=/admin/home-alexmallen/sft-logs/err-%j.out      # Set this dir where you want slurm outs to go
#SBATCH --array=0-95
#SBATCH --account=interpretability
#SBATCH --open-mode=append
#SBATCH --requeue

# Activate your environment, if needed
source /admin/home-alexmallen/.bashrc

#!/bin/bash

args_rank=$SLURM_ARRAY_TASK_ID

# Run your script
srun python /admin/home-alexmallen/elk-generalization/elk_generalization/training/run_sft.py --rank $args_rank #--weak-only
