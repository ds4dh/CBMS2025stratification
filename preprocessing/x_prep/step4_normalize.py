import glob 
import os
import sys
import pandas as pd
import numpy as np
import random
import torch
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from joblib import Parallel, delayed
from utils.config import PRIMARY_KEY_MAP, TIME_VARS_MAP_DATASET_LIST_KEYS, find_last_timestep_per_pkey, sALL_FEATURES as ALL_FEATURES

print(f"Number of cores available processes: {os.cpu_count()}")
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mimic_demo', help='dataset to use')
parser.add_argument('--num_samples', type=int, default=-1, help='number of samples to select')
parser.add_argument('--INPUT_DIR', type=str, help='Input directory', default=None)
parser.add_argument('--OUTPUT_DIR', type=str, help='Output directory', default=None)
parser.add_argument('--split_dir', type=str, help='Output directory', default=os.path.join('data','processed','splits'))

args = parser.parse_args()
num_samples = args.num_samples
dataset = args.dataset
if dataset == 'miiv':
    from utils.config import subject_to_icustay
split_dir = args.split_dir
pkey = PRIMARY_KEY_MAP[dataset]

print(f'Processing {dataset}')
dataset_name = args.dataset
if num_samples == -1:
    los_dataset = pd.read_csv(os.path.join('data', 'processed', f'step_3_{dataset_name}', f'{dataset_name}_static_los_icu.csv'))
    num_samples = len(los_dataset)
    print(f"Number of samples not provided. Using all {num_samples} samples")
dataset = dataset_name
INPUT_DIR = args.INPUT_DIR
OUTPUT_DIR = args.OUTPUT_DIR
if INPUT_DIR is None:
    INPUT_DIR = os.path.join("data", "processed", f"step_3_{dataset_name}")
if OUTPUT_DIR is None:
    OUTPUT_DIR = os.path.join("data", "processed", f"step_4_{dataset_name}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Get last timesteps per pkey
tkeys = TIME_VARS_MAP_DATASET_LIST_KEYS[dataset]
last_timesteps_per_pkey = find_last_timestep_per_pkey(INPUT_DIR, dataset, tkeys, pkey)
all_pkeys = list(last_timesteps_per_pkey.keys())
print(f"Number of patients with los: {len(all_pkeys)}")
if not all_pkeys:
    import pdb; pdb.set_trace()

selected_ids = all_pkeys

if dataset == 'miiv':
    # Split by subject id
    subject_ids = list(subject_to_icustay.keys())

    train_subjects, test_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.1, random_state=42)
    
    # Map back to ICU stays from subject IDs
    train_subjects = [subject_to_icustay[subject] for subject in train_subjects]
    val_subjects = [subject_to_icustay[subject] for subject in val_subjects]
    test_subjects = [subject_to_icustay[subject] for subject in test_subjects]
    
    # Flatten the lists
    train_subjects = [item for sublist in train_subjects for item in sublist]
    val_subjects = [item for sublist in val_subjects for item in sublist]
    test_subjects = [item for sublist in test_subjects for item in sublist]
    
    selected_ids = train_subjects + val_subjects + test_subjects
    
    if len(selected_ids) > num_samples:
        # Calculate proportions
        total_current = len(selected_ids)
        proportion_train = len(train_subjects) / total_current
        proportion_val = len(val_subjects) / total_current
        proportion_test = len(test_subjects) / total_current

        # Initial allocation based on proportions
        num_train = int(num_samples * proportion_train)
        num_val = int(num_samples * proportion_val)
        num_test = num_samples - (num_train + num_val)  # Ensure total matches

        # Ensure we do not exceed available subjects
        num_train = min(num_train, len(train_subjects))
        num_val = min(num_val, len(val_subjects))
        num_test = min(num_test, len(test_subjects))

        # Calculate remaining samples after initial allocation
        allocated = num_train + num_val + num_test
        remaining = num_samples - allocated

        if remaining > 0:
            # Determine available capacity in each split
            capacity_train = len(train_subjects) - num_train
            capacity_val = len(val_subjects) - num_val
            capacity_test = len(test_subjects) - num_test

            # Create a list of splits with their capacities
            splits = [
                ('train', capacity_train),
                ('val', capacity_val),
                ('test', capacity_test)
            ]

            # Sort splits by available capacity in descending order
            splits.sort(key=lambda x: x[1], reverse=True)

            # Distribute remaining samples
            for split_name, capacity in splits:
                if remaining == 0:
                    break
                if capacity <= 0:
                    continue
                add = min(capacity, remaining)
                if split_name == 'train':
                    num_train += add
                elif split_name == 'val':
                    num_val += add
                elif split_name == 'test':
                    num_test += add
                remaining -= add

        # Now sample without replacement, ensuring no over-sampling
        train_subjects = np.random.choice(train_subjects, num_train, replace=False).tolist()
        val_subjects = np.random.choice(val_subjects, num_val, replace=False).tolist()
        test_subjects = np.random.choice(test_subjects, num_test, replace=False).tolist()
        selected_ids = train_subjects + val_subjects + test_subjects

    print(f"Selected {len(selected_ids)} samples from {len(all_pkeys)} patients")

else:
    # Step 1: Select a subset of n_samples patients from the subject_to_icustay dictionary
    if len(selected_ids) > num_samples:
        selected_ids = np.random.choice(selected_ids, num_samples, replace=False).tolist()
    print(f"Selected {len(selected_ids)} samples from {len(all_pkeys)} patients")
    # Custom split using selected_subject_ids
    train_subjects, test_subjects = train_test_split(selected_ids, test_size=0.25, random_state=42)
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.1, random_state=42)

train = train_subjects
val = val_subjects
test = test_subjects
print('train:', len(train), 'val:', len(val), 'test:', len(test))
# Process files
ffiles = [file for file in os.listdir(f'{INPUT_DIR}')]
ffiles = [file for file in ffiles if any(feature in file for feature in ALL_FEATURES)]
os.makedirs(f'{OUTPUT_DIR}', exist_ok=True)  # Create the output directory once
print('Creating output directory:', OUTPUT_DIR)
# File processing function for parallelization
def process_file(file):
    print(f"#### Processing {file}")
    file_name = f'{INPUT_DIR}/' + file
    df = pd.read_csv(file_name)
    # Skip empty DataFrames
    if df.shape[0] == 0:
        return
    # Remove rows with NaN values
    df.dropna(inplace=True)
    # df is in train val or test
    if df.shape[0] == 0: # , f"No data for {file} after"
        import pdb; pdb.set_trace()
    try:
        # assert any of train, val, test
        assert df[pkey].isin(train + val + test).any(), f"Data not in train, val, or test for {file}"
        df = df[df[pkey].isin(train + val + test)]

    except Exception as e:
        print(f"none of sample ids have sample in {file}: {e} for miiv" )
        raise e
    # if 'ws_' in file_name: # TODO handle as seperate variables 
    #     # split 3,4 colum as differnet variable
    #     dfws1 = df[[0,1,2]]
    #     dfws2 = df[[0,1,3]]

    # Boolean and non-boolean column separation
    boolean_cols = []
    for col in df.columns:
        if col == pkey or col == 'tkey' or (df[col].nunique() == 2):
            boolean_cols.append(col)
            continue
        
        # Split DataFrame by train, val, test
        df_train = df[df[pkey].isin(train)]
        df_val = df[df[pkey].isin(val)]
        df_test = df[df[pkey].isin(test)]

        # Skip columns with no data
        if df_train.shape[0] == 0:
            continue

        # Normalize non-boolean columns
        normalizer = RobustScaler()
        # try:
        normalizer.fit(df_train[col].values.reshape(-1, 1))
        df[col] = normalizer.transform(df[col].values.reshape(-1, 1))
    # Save processed DataFrame
    output_file = f'{OUTPUT_DIR}/' + file
    print(f"Saving {output_file}")
    df.to_csv(output_file, index=False)

assert len(ffiles) > 0, f"No files found in {INPUT_DIR}"
if sys.platform == 'darwin':
    Parallel(n_jobs=1)(delayed(process_file)(file) for file in ffiles)
else:
    Parallel(n_jobs=-1)(delayed(process_file)(file) for file in ffiles)
    # for file in ffiles:
    #     process_file(file)

split_dataset_dir = os.path.join(split_dir, f'split_{dataset}')
os.makedirs(split_dataset_dir, exist_ok=True)
# Save train, val, test splits
fpath = os.path.join(split_dataset_dir, f'train.pkl')
print(f"Saving train split to {fpath}")
with open(fpath, 'wb') as f:
    pickle.dump(train, f)
fpath = os.path.join(split_dataset_dir, f'val.pkl')
print(f"Saving val split to {fpath}")
with open(fpath, 'wb') as f:
    pickle.dump(val, f)
fpath = os.path.join(split_dataset_dir, f'test.pkl')
print(f"Saving test split to {fpath}")
with open(fpath, 'wb') as f:
    pickle.dump(test, f)

# dump selected_ids
fpath = os.path.join(split_dataset_dir, f'selected_ids.pkl')
print(f'Saving selected ids to {fpath}')
with open(fpath, 'wb') as f:
    pickle.dump(selected_ids, f)