## CHANGELOG
### Use it to document your changes in your branch

## [Unreleased]
### Added
- `fine_tune_main.py`, `fine_tune.py`, `data_extraction.py`, `data_extraction.ipynb`.
- fine_tune* allows the possibility to fine tune existing geomodels.
- data_extraction* are scripts to collect annotations from iNatAtor database.
- added example annotation data to use fine-tuner out of the box `data/annotations/example.csv`
- added `.env.copy` a skeleton `.env` to store database secrets
- added `scripts` to have sbatch jobs ready to run

### Changed
- `README.md` updated with instructions on fine tuning
- `losses.py` added 2 new loss functions that are used in fine tuning

## 6/3/2024 v1.0.0
### Added
- CHANGELOG.md to keep track of progress and changes to repo
### Changes
## 6/3/2024 v1.0.1
### Added
- Created CHANGELOG.md, we should create one in default branch as well
- reproduce.py and reproduce_tables.ipynb
- These two files allow you to reproduce and analyze results from the SINR paper
- Added instructions on how to run the files at the end of README.md, please read it before attempting to run
### Changes
- Changed readme
- changed a line in viz_ica.py that had incorrect path leading to error, this should be carried to another patch branch
- deleted slurm scripts and unnecessary parts of the commit