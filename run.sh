#!/bin/bash
#
# all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
# set a job name
#SBATCH --job-name=doccvqa_baselines
#################
# working directory
#SBATCH -D /home/rperez/Baselines/
##############
# File for job output, you can check job progress
#SBATCH --output=/home/rperez/Pythia/slurm/%j.out
#################
# File for errors from the job
#SBATCH --error=/home/rperez/Pythia/slurm/%j.err
#################
# Time you think you need
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the # faster your job will run.
#SBATCH --time=00:00:00
#################
# --gres will give you one GPU, you can ask for more, up to 4 (or how ever many are on the node/card)
# 1080Ti, TitanXp
#SBATCH --gres gpu:4
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
#SBATCH --mem=40000
#################
#SBATCH --export=ALL


################# Prepare the experiment to run #################

# Train SingleDocVQA
# CODE="python -m torch.distributed.launch --nproc_per_node 4 --master_port=9901 tools/run.py --tasks vqa --dataset t5_singledocvqa --model layout_t5_base --config configs/vqa/docvqa/layoutt5/layoutt5_singledocvqa_1024_textract_ec2.yml --save_dir /data3fast/users/rperez/pythia_save/no_pretrain --distributed True"

# Finetune SingleDocVQA with Textract OCR
# This is done!!
# CODE="python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks vqa --dataset t5_singledocvqa --model layout_t5_base --config configs/vqa/docvqa/layoutt5/layoutt5_singledocvqa_ft_1024_textract_ec2.yml --resume_file /data3fast/users/rperez/t5_weights/layoutt5_pt_16x1024_lr2e4/model_31000.ckpt --save_dir /data3fast/users/rperez/pythia_save/textract --distributed True"

# Train on DocCVQA
# CODE="python -m torch.distributed.launch --nproc_per_node 4 --master_port=9901 tools/run.py --tasks vqa --dataset t5_doccvqa --model layout_ct5_base --config configs/vqa/docvqa/layoutct5/layoutct5_singledocvqa_1024_ec2.yml --resume_file /data3fast/users/rperez/t5_weights/layoutt5_pt_16x1024_lr2e4__ft_16x1024_lr2e4_textract/model_29000.ckpt  --save_dir /data3fast/users/rperez/pythia_save/doccvqa_10CLS --distributed True"

# Train on DocCVQA Collection-wise
CODE="python -m torch.distributed.launch --nproc_per_node 4 --master_port=9901 tools/run.py --tasks vqa --dataset t5_doccvqa --model layout_ct5_base --config configs/vqa/docvqa/layoutct5/layoutct5_singledocvqa_1024_ec2.yml --resume_file /data3fast/users/rperez/t5_weights/layoutt5_pt_16x1024_lr2e4__ft_16x1024_lr2e4_textract/model_29000.ckpt  --save_dir /data3fast/users/rperez/pythia_save/doccvqa_collection_10CLS --distributed True"



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
