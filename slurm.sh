#!/bin/bash -l
#SBATCH -J KMC_SUBMISSION
#SBATCH -o output/%x_%A.out
#SBATCH -e output/%x_%A.err
#SBATCH -t 00:30:00
#SBATCH -p tier3 -A heteroxide
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2000

set -e

rm -rf output
mkdir -p output

# 1. Load your standard CPU environments
spack load lammps@20240207 /k7xjs3x

# 3. Run the heavy NEB calculations (Uses ALL cores, partitioned internally)
# To submit the NEB array and automatically compile KMC_Barriers.csv afterward,
# run submit_neb_with_compile.sh instead of submitting this file directly.
srun python3 -u -m KMC KMC Barrier/KMC_Barriers.csv 2000000 -v 2782 5235 6234 11652 6514 -T 1000 -k 8.61733e-5 --log logs/benchmark_1000K.log