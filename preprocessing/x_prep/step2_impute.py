import pandas as pd
from tqdm import tqdm
import glob 
import shutil
import argparse
from sklearn.impute import KNNImputer
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import numpy as np
import random
from utils.config import PRIMARY_KEY_MAP, TIME_VARS_MAP_DATASET_LIST_KEYS, find_last_timestep_per_pkey, ALL_FEATURES

np.random.seed(42)
random.seed(42)

print('Time start: ', pd.Timestamp.now())

def fix_feature_death_hirid_sic_miiv(INPUT_DIR, dataset_name):
    ''' Ensure consistency by adding a death column and filling missing data '''
    
    if dataset_name == 'hirid':
        return
    fpath = os.path.join(INPUT_DIR, f'{dataset_name}_ts_death.csv')
    df = pd.read_csv(fpath)
    # Fill missing death data: if value in OffsetOfDeath is NaN, set death to 0, otherwise 1
    if dataset_name == 'sic':
        df['death'] = df['OffsetOfDeath'].apply(lambda x: 0 if pd.isna(x) else 1)

    return fix_for_missing_keys(df, 'death')


def fix_for_missing_keys(df, feature):    
    """
    Steps: 
    - Finds all pkeys in dataset
    """
    # Get primary and time-related keys
    pkey = PRIMARY_KEY_MAP[dataset_name]
    tkeys = TIME_VARS_MAP_DATASET_LIST_KEYS[dataset_name]
    
    # Find the last timestep per pkey
    los_per_pkey = find_last_timestep_per_pkey(INPUT_DIR, dataset_name, tkeys, pkey_column=pkey)
    pkeys = list(los_per_pkey.keys())
    
    # for missing keys add new rows to the feat8re 0 with time value the los and pky the pky 
    # chosen tk 
    tkey_chosen =[tvar for tvar in TIME_VARS_MAP_DATASET_LIST_KEYS[dataset_name] if tvar in df.columns][0]
    missing_keys = [pkey_id for pkey_id in pkeys if pkey_id not in df[pkey].values]
    # Get the unique values of the feature column
    unique_values = df[feature].unique()
    
    # Check if there is only one unique value
    if len(unique_values) == 1:
        existing_feature_value = unique_values[0]
        if existing_feature_value == 0:
            to_fill = 1 # if not mentioned it is the opposite of the existing value i.e. existence death 
        elif existing_feature_value == 1:
            to_fill = 0
            
        if missing_keys: 
            missing_df = pd.DataFrame({
                pkey: missing_keys,
                tkey_chosen: [los_per_pkey[k] for k in missing_keys],
                feature: [to_fill] * len(missing_keys)
            })
                
            # Concatenate the original DataFrame with the new rows
            df = pd.concat([df, missing_df], ignore_index=True)
    # Sort by pkey and tkeys to maintain consistency
    df = df.sort_values(by=[pkey] + [tkey_chosen])
    
    # Ensure no missing keys
    missing_keys = [pkey_id for pkey_id in pkeys if pkey_id not in df[pkey].values]
    assert len(missing_keys) == 0, "There should be no missing keys"
    # Save the fixed dataset
    df.to_csv(f'{OUTPUT_DIR}/{dataset_name}_ts_{feature}.csv', index=False)
    return df


def stays_from_same_subject_miiv(INPUT_DIR, dataset_name):
    ''' Group ICU stays by subject ID '''
    df = pd.read_csv(f'{INPUT_DIR}/{dataset_name}_icustays.csv')
    map_subject_stayid = df.groupby('subject_id')['stay_id'].apply(list).to_dict()
    return map_subject_stayid


def knn_impute_df(all_stats:np.array, k:int=1):
    ''' Impute missing stay information using KNN '''
    imputer = KNNImputer(n_neighbors=k)
    i_all_stats = imputer.fit_transform(all_stats)
    return i_all_stats

# Function to process a single file in chunks
def process_file(file, dataset, blacklist):
    chunk_size = 100000  # Define the size of chunks to process
    pkey_col = PRIMARY_KEY_MAP[dataset]
    
    with pd.read_csv(file, chunksize=chunk_size) as reader:
        # Open file for writing (write mode), but we won't write until we filter out unwanted rows
        with open(file, 'w', newline='') as f_out:
            writer = None
            for chunk in reader:
                # Filter out rows where pkey is in the blacklist
                chunk_filtered = chunk[~chunk[pkey_col].isin(blacklist)]
                
                # Write the filtered chunk to CSV
                if writer is None:
                    # If this is the first chunk, write the header
                    chunk_filtered.to_csv(f_out, index=False, header=True)
                    writer = True
                else:
                    # For subsequent chunks, skip the header
                    chunk_filtered.to_csv(f_out, index=False, header=False)


def read_file_group(file,pkey, data_dict, feature_existence):
    df = pd.read_csv(file)
    group = df.groupby(pkey)
    feature_name = os.path.basename(file).replace(".csv", "")  # Extract the feature name
    for group_key, data in group:
        if group_key not in data_dict:
            data_dict[group_key] = {}
            feature_existence[group_key] = []
        data_dict[group_key][feature_name] = data
        feature_existence[group_key].append(len(data) > 0)
        if 'tkey' in data.columns:
            try:
                data = data.sort_values(by='tkey')  # Sort by timestep if exists
            except:
                import pdb; pdb.set_trace()
    return data_dict, feature_existence


def blacklist_data(input_folder, output_folder, dataset, threshold=0.5, file_pattern="*.csv"):
    full_pattern = os.path.join(input_folder, file_pattern)
    print(f"Looking for files matching: {full_pattern}")
    files = sorted(glob.glob(full_pattern))
    print(f"Number of files: {len(files)}")
    pkey = PRIMARY_KEY_MAP[dataset]
    data_dict, feature_existence = {}, {}
    
    for file in tqdm(files):
        data_dict, feature_existence = read_file_group(file, pkey, data_dict, feature_existence)
    
    pct_feature_existence = {pkey: len(features) / len(ALL_FEATURES) for pkey, features in feature_existence.items()}
    
    # Blacklist pkeys that have less than 50% of the features existing this is 
    blacklist = [pkey for pkey, pct in pct_feature_existence.items() if pct <= 0.5]
    # blacklist = []
    
    print(f"Blacklisted pkeys: {blacklist}")
    # Sequential processing of files (no parallelism)
    for file in files:
        df = pd.read_csv(file)
        # remove rows where pkey has value in blacklist
        shape_before = df.shape
        df = df[~df[pkey].isin(blacklist)]
        shape_after = df.shape
        output_file = file.replace(input_folder, output_folder)
        shape_after = df.shape
        dif = shape_before[0] - shape_after[0]
        df.to_csv(output_file, index=False)
        print('\ninput_file:', file)
        print('shape before:', shape_before)
        print('shape after:', df.shape)
        print('removed in total :', dif)
        print('output_file:', output_file)
    return data_dict, pct_feature_existence, blacklist


def analyze_df(feature_path: str): 
    is_static = 'static' in feature_path
    is_time_series = 'ts' in feature_path
    is_ws = 'ws' in feature_path
    split_pattern = 'ts' if is_time_series else 'static' if is_static else 'ws'
    return is_static, is_time_series, is_ws


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic_demo', help='Dataset name')
    args = parser.parse_args()
    dataset_name = args.dataset
    dataset = dataset_name
    INPUT_DIR = os.path.join("data", "raw", f"step_1_{dataset_name}")
    OUTPUT_DIR = os.path.join("data", "processed", f"step_2_{dataset_name}")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    # data/processed/step_1_mimic_demo
    print('creating output dir:', OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shutil.copytree(INPUT_DIR, OUTPUT_DIR, dirs_exist_ok=True)
    data_dict, pct_feature_existence, blacklist= blacklist_data(INPUT_DIR, OUTPUT_DIR, dataset_name, threshold=0.5,file_pattern="*.csv")
    
    pkey_col = PRIMARY_KEY_MAP[dataset]
    # overide fiels needed to be iomputed in OUTPUTDIR - we have coppied all remember?
    for feature_path in os.listdir(INPUT_DIR):
        if 'icds' in feature_path: # Dont keep icds for next steps 
            continue 
        is_static, is_time_series, is_ws = analyze_df(feature_path)
        df = pd.read_csv(os.path.join(INPUT_DIR, feature_path))
        # feature_col_name # 3rd column is the feature name
        feature_col_name = list(df.columns)[-1]
        if 'static_los.csv' in feature_path and dataset == 'sic':
            feature_path=feature_path.replace('los', 'los_icu') # ricu  incosistency in los anming for sic
        # remove rwos where pkey_col has value in blackkeys
        df = df[~df[pkey_col].isin(blacklist)]
        if df.shape[0] < 2:
            continue

        if 'death' in feature_path:
            df = fix_feature_death_hirid_sic_miiv(INPUT_DIR, dataset_name)
        if df[feature_col_name].nunique() == 1 and len(df.columns)<4: # <4 for ws
            print(f"Feature {feature_col_name} has only one unique value. Imputing.")
            df = fix_for_missing_keys(df, feature_col_name)
        df.to_csv(os.path.join(OUTPUT_DIR, feature_path), index=False)
print('Time end: ', pd.Timestamp.now())
