# Python  repository for ICU-TSB: A Benchmark for Temporal Patient Representation Learning for Unsupervised Stratification into Patient Cohorts 
-  Submitted and accepted in CBMS2025

You need to download and have set up Physionet access (for which you need the CITI certificate), for each fo the ICU datasets (MIMIC-IC, eICU and SiCDB) with rICU preprocessing https://github.com/eth-mds/ricu (required R). 


# TLDR instructions with the MIMIC-III-demo dataset
## Programming Language  requirements 
-  Rscript --version 
Rscript  (R) version 4.3.2 (2023-10-31)
Note:  unfortunatly the `units` R package library is only working in Windows hence at least for the `step_1.R` script you will need to Windows.
The rest (everything apart from `step1.r`) should be run in Linux or MacOSX.
That said we are actively  working in rewritiing the library in python/linux using a fork of rICU.
-  python --version
Python 3.10.12

## data
Rscript preprocessing/x_prep/step1.r --dataset mimic_demo
python -m preprocessing.x_prep.step2_impute --dataset mimic_demo 
python -m preprocessing.x_prep.step3_encoding --dataset mimic_demo
python -m preprocessing.x_prep.step4_normalize --dataset mimic_demo
python -m preprocessing.x_prep.step5_group --dataset mimic_demo

# models
- STAT
python -m models.stats --dataset mimic_demo
- singleLSTM
python -m models.lstmv5 --dataset mimic_demo --mode train --max_steps 10000 --max_patients 10000 --learning_rate 5e-5 --batch_size 2 --timeseries_model singleLSTM
- GRU
python -m models.lstmv5 --dataset mimic_demo --mode train --max_steps 10000 --max_patients 10000 --learning_rate 5e-5 --batch_size 2 --timeseries_model singleLSTM --gru

# stratification
- STAT
python -m experiments.IR12 --dataset mimic_demo --trials 10 --optimize --mode stats --fpath
data/embeddings/stats_mimic_demo/stats_test_mimic_demo_patient_embeddings.csv
-  GRU
python -m experiments.IR12 --dataset mimic_demo --trials 10 --optimize --mode gru --fpath data/embeddings/gru_train_mimic_demo_e_10_ms_10000_samples_10000__bs_2.shelve

- LSTM
python -m experiments.IR12 --dataset mimic_demo --trials 10 --optimize --mode lstm --fpath data/embeddings/singleLSTM_train_mimic_demo_e_10_ms_10000_samples_10000__bs_2.shelve       



