## biobank bmrc pipeline
* This repo contains code for extracting imaging data from bmrc for bigflica, and training classifiers based on the results

### python environment
* To set up python environment to run code in this repo on bmrc, use `module load Python/3.7.4-GCCcore-8.3.0` version of python
* Install `requirements.txt` in a virtualenv, following [bmrc instructions](https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/python-on-the-bmrc-cluster) to get both skylake and ivybridge versions working
* `source xxx-skylake/bin/activate` to activate environment before running any of the code
* If sending jobs, `sub_lines.sh` shows an example of how to do this, modify computing requirements based on job load. Node limits are listed in [bmrc webpage](https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/cluster-usage#cluster-queues-and-nodes).

### extracting imaging data
* Given that the non-imaging data extracted from biobank using existing study number is stored on jalapeno, currently the code are split into 2 repos. It's better to extract relevant eids from japapeno, transfer those lists to bmrc, and do relevant imaging analyses on bmrc
* Put jalapeno extracted eids in `subjs` as csv files
* Run `match_bridge.py` to convert eids to bmrc ones using bridge file, new eids are stored in `bmrc_subjs` directory
* Run `check_mods.py` to get eids that have all imaging modalities, remaining eids are stored in `bmrc_full` directory
* Run `combine_labels.py` to combine bmrc and japapeno eids, results stored in `labels_full` directory

### running extraction jobs
* When `bmrc_full` eids are ready, use `sub_lines.sh` to submit extraction jobs, where it uses functions in `extract_mod` that might require modification. Depending on the number of subjects, this can take a few days
* When all data is extracted, run bigflica using code in repo [bfl_pain](https://github.com/syzhang/bfl_pain) (in a seperate repo because it requires a different python2 environment)

### classifier training and testing
* `qsidp` directory contains questionnaire and IDP data copied over from jalapeno (use sftp to move IDP data from japapeno to bmrc if needed)
* Notebook `dev_paincontrol.ipynb` shows the pipeline development for pain vs no pain control classification
* Notebook `dev_paintype.ipynb` contains pipeline for disease classifcation 
* Notebook `dev_painquestion.ipynb` shows the pipeline development for pain questionnaire based classification
