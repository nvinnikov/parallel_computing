#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --time=00:01:00
#SBATCH --output=graph_traversal_%j.log

echo "Date = $(date)"
echo "Host = $(hostname -s)"
echo "WorkDir = $(pwd)"
echo ""
echo "Number of Nodes Allocated = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated = $SLURM_NTASKS"
echo "Numbers of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module load CUDA

srun /home/nvinnikov/graph > graph_output.log
