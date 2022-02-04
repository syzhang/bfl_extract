## biobank bmrc pipeline

### directory
* `bmrc_subjs` contains bmrc subject eids converted using bridge file
* `labels_full` contains jalapeno subject eids with appropriate labels
* `qsidp` contains questionnaire and IDP data copied over from jalapeno

### code (relevant ones)
* `check_mods` contains list of imaging modalities used for bigflica
* `extract_mod` contains functions of extracting bmrc imaging data to npys
* `match_bridge` matches jalapeno subject eids to bmrc ones
* `dev_paincontrol.ipynb` notebook contains pipeline for pain vs no pain control classification
* `dev_paintype.ipynb` notebook contains pipeline for disease classifcation 
