import glob
import os
import pandas as pd
import numpy as np
import random
import torch
import argparse
from tqdm import tqdm
import pickle
import logging
from utils.config import TIME_VARS_MAP_DATASET_LIST_KEYS, PRIMARY_KEY_MAP

# python versoin print 
import sys 
# print datetime padnas start msg 
print('Start')
print (pd.Timestamp.now())	
print("Python Version: ", sys.version)
# pandas 
print("Pandas Version: ", pd.__version__)
# numpy
print("Numpy Version: ", np.__version__)
# torch
print("Torch Version: ", torch.__version__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. **Environment Setup**
    logging.info(f"Number of available cores: {os.cpu_count()}")
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # 2. **Argument Parsing**
    parser = argparse.ArgumentParser(description="Process dataset and consolidate into a pickle file.")
    parser.add_argument('--dataset', type=str, default='mimic_demo', help='Dataset to use')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to select (-1 for all)')
    parser.add_argument('--INPUT_DIR', type=str, help='Input directory', default=None)
    parser.add_argument('--OUTPUT_DIR', type=str, help='Output directory', default=None)
    parser.add_argument('--SPLIT_DIR_BASE', type=str, help='Base directory for splits', default=os.path.join('data', 'processed', 'splits'))
    parser.add_argument('--folder_save', type=bool, help='Save the processed data in a folder structure', default=False)
    
    args = parser.parse_args()

    # 3. **Set Variables**
    dataset = args.dataset
    num_samples = args.num_samples
    tkeys = TIME_VARS_MAP_DATASET_LIST_KEYS.get(dataset, [])
    
    # Handle num_samples in OUTPUT_DIR name
    num_samples_str = 'all' if num_samples == -1 else str(num_samples)
    
    # Define INPUT_DIR and OUTPUT_DIR with num_samples in the name
    INPUT_DIR = args.INPUT_DIR or os.path.join("data", "processed", f"step_4_{dataset}")
    OUTPUT_DIR = args.OUTPUT_DIR or os.path.join("data", "processed", f"step_5_{dataset}_{num_samples_str}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 4. **Load Primary Key and Selected IDs**
    pkey = PRIMARY_KEY_MAP.get(dataset)
    assert pkey is not None, f"Primary key for dataset '{dataset}' not found."

    split_dir = os.path.join(args.SPLIT_DIR_BASE, f'split_{dataset}')
    selected_ids_path = os.path.join(split_dir, 'selected_ids.pkl')
    assert os.path.exists(selected_ids_path), f"Selected IDs file not found at {selected_ids_path}"
    
    with open(selected_ids_path, 'rb') as f:
        selected_ids = pickle.load(f)
    
    # If num_samples is not -1, limit the selected_ids
    if num_samples != -1:
        selected_ids = selected_ids[:num_samples]
    
    # 5. **Initialize Data Storage**
    processed_data = {}

    # 6. **Define Processing Function**
    def process_and_store_files(input_folder, file_pattern):
        full_pattern = os.path.join(input_folder, file_pattern)
        files = sorted(glob.glob(full_pattern))
        
        if not files:
            logging.error(f"No files found for pattern {full_pattern}")
            raise ValueError(f"No files found for pattern {full_pattern}")
        
        for file in tqdm(files, desc=f"Processing files matching {file_pattern}"):
            try:
                for chunk in pd.read_csv(file, chunksize=100000):
                    grouped = chunk.groupby(pkey)
                    for group_key, data in grouped:
                        if group_key not in selected_ids:
                            continue
                        
                        # Remove the primary key column
                        data = data.drop(columns=[pkey])
                        
                        # Sort by 'tkey' if it exists
                        if 'tkey' in data.columns:
                            data = data.sort_values(by='tkey')
                        
                        # Initialize nested dictionary if necessary
                        if group_key not in processed_data:
                            processed_data[group_key] = {}
                        
                        # Determine the file key (e.g., 'static', 'ts')
                        file_key = os.path.basename(file).replace(".csv", "")
                        
                        # Initialize DataFrame for this file_key if necessary
                        if file_key not in processed_data[group_key]:
                            processed_data[group_key][file_key] = data.copy()
                        else:
                            # Concatenate the new data
                            processed_data[group_key][file_key] = pd.concat([processed_data[group_key][file_key], data], ignore_index=True)
            except Exception as e:
                logging.error(f"Error processing {file}: {e}")
                # append to txt the unprocessed files
                # dataset from filepattern
                dataset = file_pattern.split("_")[0]
                with open(f'{OUTPUT_DIR}/step_5_{dataset}_unprocessed_files.txt', 'a') as f:
                    f.write(f"{file}\n")
                continue
    
    # 7. **Process Static and Time Series Files**
    process_and_store_files(INPUT_DIR, "*static_*.csv")
    process_and_store_files(INPUT_DIR, "*ts_*.csv")
    # Uncomment the line below if window series files need to be processed
    # process_and_store_files(INPUT_DIR, "*ws_*.csv")
    
    # 8. **Process ICD Mappings if File Exists**
    icds_file = os.path.join(INPUT_DIR, f'{dataset}_icds.csv')
    if os.path.exists(icds_file):
        icds = pd.read_csv(icds_file)
        grouped_icds = icds.groupby(pkey)
        for group_key, data in tqdm(grouped_icds, desc="Processing ICD mappings"):
            if group_key not in selected_ids:
                continue
            # Initialize nested dictionary if necessary
            if group_key not in processed_data:
                processed_data[group_key] = {}
            # Store ICD data
            processed_data[group_key]['icds'] = data.drop(columns=[pkey])

    # 9. **Save All Processed Data into a Single Pickle File**
    output_pickle_path = os.path.join(OUTPUT_DIR, f'{dataset}_processed_data.pkl')
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # save folder version of the processed data  of --folder_save is passed as true 
    if args.folder_save:
        for group_key, data in processed_data.items():
            group_folder = os.path.join(OUTPUT_DIR, str(group_key))
            os.makedirs(group_folder, exist_ok=True)
            for file_key, df in data.items():
                file_path = os.path.join(group_folder, f'{file_key}.csv')
                logging.info(f"Saving {file_key} data for {group_key} to {file_path}")
                df.to_csv(file_path, index=False)
    logging.info(f"All data has been processed and saved to {output_pickle_path}")

if __name__ == "__main__":
    main()
# python -m preprocessing.x_prep.step5_group --dataset sic
# python -m preprocessing.x_prep.step5_group --dataset miiv
# python -m preprocessing.x_prep.step5_group --dataset eicu
# python -m preprocessing.x_prep.step5_group --dataset mimic_demo