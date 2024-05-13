#!/bin/bash
#SBATCH --export=ALL
#SBATCH --nodes=1 
#SBATCH --time=00:30:00 
#SBATCH --cpus-per-task=2 
#SBATCH --gres=gpu:kepler:1

for ((i = 0; i <= 26; i++)); do 
  time ./sortingNetworks $i
done
