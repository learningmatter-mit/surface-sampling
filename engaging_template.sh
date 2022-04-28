#!/bin/bash
#SBATCH -p sched_mit_rafagb
#SBATCH -J jtprop
#SBATCH -o jtprop_train_filtered-%j.out
#SBATCH -t 10-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=300gb
#SBATCH --gres=gpu:1

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
cat $0
echo ""

#module load cuda/10.2
module load anaconda3/2020.11
source activate mlenv

python pipeline/automation.py jtprop --train --validate