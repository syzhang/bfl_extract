#!/bin/bash

#$ -q short.qc@@short.hga
#$ -pe shmem 1
#$ -o /well/seymour/users/uhu195/python/shoes/
#$ -e /well/seymour/users/uhu195/python/shoes/

# note that you must load whichever main Python module you used to create your virtual environments before activating the virtual environment
module load Python/3.7.4-GCCcore-8.3.0

# Activate the ivybridge or skylake version of your python virtual environment
# NB The environment variable MODULE_CPU_TYPE will evaluate to ivybridge or skylake as appropriate
source /well/seymour/users/uhu195/python/extract-py3.7.4-${MODULE_CPU_TYPE}/bin/activate

# continue to use your python venv as normal
echo "Working on IC number $1"
# python /well/seymour/users/uhu195/python/extract_npy/compare_painquestion.py $1 qs
# python /well/seymour/users/uhu195/python/extract_npy/compare_painquestion.py $1 idp
# python /well/seymour/users/uhu195/python/extract_npy/compare_painquestion.py $1 qsidp
# python /well/seymour/users/uhu195/python/extract_npy/compare_painquestion.py $1 nobfl

# python /well/seymour/users/uhu195/python/extract_npy/compare_paincontrol.py $1 qs
# python /well/seymour/users/uhu195/python/extract_npy/compare_paincontrol.py $1 idp
# python /well/seymour/users/uhu195/python/extract_npy/compare_paincontrol.py $1 qsidp
python /well/seymour/users/uhu195/python/extract_npy/compare_paincontrol.py $1 nobfl

# python /well/seymour/users/uhu195/python/extract_npy/compare_paintype.py $1 qs
# python /well/seymour/users/uhu195/python/extract_npy/compare_paintype.py $1 idp
# python /well/seymour/users/uhu195/python/extract_npy/compare_paintype.py $1 qsidp
# python /well/seymour/users/uhu195/python/extract_npy/compare_paintype.py $1 nobfl

