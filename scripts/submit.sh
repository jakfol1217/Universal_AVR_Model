#!/usr/bin/env bash

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user= # TODO: create mail
#SBATCH --constraint=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --gpus=1 # TODO: Maybe prepare scipts to work in both single- and multi-gpu mode
#SBATCH --time=0-12:00:00
#SBATCH --partition=short # TODO: Dynamically change based on time
#SBATCH --export=ALL,HYDRA_FULL_ERROR=1
#SBATCH --account=mandziuk-lab

# TODO: Setup submitit hydra plugin and check if it works
date "+%Y-%m-%d %H:%M:%S"
echo "SLURMD_NODENAME: ${SLURMD_NODENAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Command: python ${1}" "${@:2}"
echo "Enroot version: $(enroot version)"
enroot start \
    --rw \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    -m /home2/faculty/akaminski/Universal_AVR_Model/src:/app/src:ro:x-create=dir,bind \
    -m /home2/faculty/akaminski/Universal_AVR_Model/.env:/app/.env:ro:x-create=file,bind \
    -m /etc/slurm:/etc/slurm \
    universal-avr-system-latest \
    "${1}" "${@:2}"
echo "Enroot exited with code: $?"
date "+%Y-%m-%d %H:%M:%S"
