#!/bin/bash
#SBATCH -n 1 #Request 4 tasks (cores)
#SBATCH -N 1 #Request 1 node
#SBATCH -C centos7 #Request only Centos7 nodes
#SBATCH -p sched_mit_rafagb
#SBATCH --time=01:00:00
#SBATCH --mem=100G #MaxMemPerNode

exp_name='test'
source activate mlenv
python GaN_0001_multiple_runs.py --runs 300 1> output.txt 2> error.txt

# python run_ala_test_mc.py -logdir ../exp/mctest_1500_10 -dataset dipeptide -device 0 -n_cgs 6 \
#                 -batch_size 32 -nsamples 20 -ndata 500 -nepochs 600 -nevals 5 -atom_cutoff 8.5 -cg_cutoff 9.5 \
#                 -nsplits 5 -beta 0.05 -activation swish -dec_nconv 5 -enc_nconv 4 -lr 0.00008 -n_basis 600 \
#                 -n_rbf 8 --graph_eval -gamma 25.0 -eta 0.0 -kappa 0.0 -patience 15 -cg_method newman -edgeorder 2 \
#                 --tqdm_flag \
#                 1> output_$exp_name-$SLURM_JOBID.txt 2> error_$exp_name-$SLURM_JOBID.txt
