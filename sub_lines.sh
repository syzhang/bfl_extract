#!/bin/bash

#$ -q short.qc@@short.hga
#$ -pe shmem 10
#$ -o /well/seymour/users/uhu195/python/shoes/
#$ -e /well/seymour/users/uhu195/python/shoes/

# note that you must load whichever main Python module you used to create your virtual environments before activating the virtual environment
module load Python/3.7.4-GCCcore-8.3.0

# Activate the ivybridge or skylake version of your python virtual environment
# NB The environment variable MODULE_CPU_TYPE will evaluate to ivybridge or skylake as appropriate
source /well/seymour/users/uhu195/python/extract-py3.7.4-${MODULE_CPU_TYPE}/bin/activate

# continue to use your python venv as normal
echo "Working on modality $1"
python /well/seymour/users/uhu195/python/extract_npy/extract_mod.py $1 
# python /well/seymour/users/uhu195/python/extract_npy/extract_mod.py 0 300

# sub loop
# for mod_num in {0..20}
# do
# qsub -pe shmem 5 extract_npy/sub_lines.sh $mod_num
# done
