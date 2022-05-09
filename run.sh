#!/bin/bash
#
# all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
# set a job name
#SBATCH --job-name=doccvqa_baselines
#################
# working directory
#SBATCH -D /home/rperez/DocCVQA_Baselines/
##############
# File for job output, you can check job progress
#SBATCH --output=/home/rperez/DocCVQA_Baselines/slurm/%j.out
#################
# File for errors from the job
#SBATCH --error=/home/rperez/DocCVQA_Baselines/slurm/%j.err
#################
# Time you think you need
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the # faster your job will run.
#SBATCH --time=00:00:00
#################
# --gres will give you one GPU, you can ask for more, up to 4 (or how ever many are on the node/card)
# 1080Ti, TitanXp
#SBATCH --gres gpu:1
# We are submitting to the batch partition
#SBATCH -p dag
#################
# Number of cores
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
# Ensure that all cores are on one machine
#SBATCH -N 1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=25000
#################
#SBATCH --export=ALL


################# Prepare the experiment to run #################

# Train SingleDocVQA
CODE="python train.py"



####################################################################

# Prepare the experiment to run
#echo "Experiment number: $1"
echo $CODE

#################
# Run the experiment
srun $CODE

#################
# Report that it has finished
echo "Done."

#free -h
