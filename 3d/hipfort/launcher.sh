#!/bin/bash

#SBATCH --partition=eap
#SBATCH --account=project_462000007 
#SBATCH --time=02:10:10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=8
#SBATCH -e slurm.err
#SBATCH -o slurm.out
#SBATCH --exclusive

source /users/vitaliem/prova_hip/heat-equation/3d/hipfort/modules
make 
srun /users/vitaliem/prova_hip/heat_eq/3d/hipfort/heat_hipfort
