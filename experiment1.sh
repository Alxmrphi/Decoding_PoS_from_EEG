#!/bin/bash
#SBATCH --time 200:0
#SBATCH --qos castles
#SBATCH --account zumerj01
#SBATCH --mem 50G
#SBATCH --nodes 1
#SBATCH --cores 1
#SBATCH --cpus-per-task=1

set -e

module purge;
module load bluebear
module load bear-apps/2019b
module load sklearn-crfsuite/0.3.6-foss-2019a-Python-3.7.2

python experiment1.py -n_avg ${1} -window_size ${2} -category ${3} -folder ${4} -ica_type ${5} -bc_type ${6} -seed ${7}
