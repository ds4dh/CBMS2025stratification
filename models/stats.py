import pickle
import joblib
from sklearn.impute import KNNImputer
import os
import pandas as pd
import numpy as np
from utils.config import categorical_features, ALL_FEATURES
from tqdm import tqdm
import random
import torch

# Seed for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

functions = ['mean', 'std', 'min', 'max', 'count']
window_names = ['fq', 'fh', 'f', 'lh', 'lq']

def calc_stats_embedding_on_timewindow(time_window: pd.Series, window_name) -> pd.Series:
    """Calculate statistics on a time window."""
    if time_window.empty:
        return pd.Series()
    try:
        stats_df = time_window.agg(functions)
    except Exception as e:
        print(f"Error calculating stats for time window: {e}")
    stats_df.index = [f"{window_name}_{col}" for col in stats_df.index]
    return stats_df

def split_timeseries_windows(df_input: pd.DataFrame) -> pd.Series:
    """Split time series data into windows and calculate statistics."""
    if df_input.shape[0] < 5:
        pad_size = 5 - df_input.shape[0]
        empty_rows = pd.DataFrame(np.nan, index=np.arange(pad_size), columns=[df_input.name])
        df = pd.concat([df_input, empty_rows], axis=0).ffill()
    else:
        df = df_input

    splits = [
        df.iloc[:len(df) // 4],
        df.iloc[:len(df) // 2],
        df,
        df.iloc[len(df) // 2:],
        df.iloc[len(df) // 4:]
    ]
    
    all_stats = []
    for window, window_name in zip(splits, window_names):
        stats = calc_stats_embedding_on_timewindow(window, window_name)
        all_stats.append(stats)

    return pd.concat(all_stats)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic_demo', help='Dataset name')
    parser.add_argument('--mode', type=str, default='test', help='Mode: train or test')
    parser.add_argument('--threshold', type=int, default=-1, help='Number of patients to process')
    args = parser.parse_args()

    dataset = args.dataset
    mode = args.mode
    threshold = args.threshold

    # Load processed data dictionary and test IDs
    input_pickle_path = f"data/processed/step_5_{dataset}_all/{dataset}_processed_data.pkl"
    with open(input_pickle_path, 'rb') as f:
        patient_data = pickle.load(f)

    test_ids_path = f"data/processed/splits/split_{dataset}/test.pkl"

    # test ids are in the keys of miiv_grouped
    import json 
    miivg= json.load(open('data/processed/miiv_grouped.json','rb'))
    # intersection patient_data.keys() and miiv_grouped.keys()
    intersection = set(patient_data.keys()).intersection(set(miivg.keys()))
    # intersection patient_data.keys() and miiv_grouped.values() flattened 
    is1 = set([int(p) for p in set(pickle.load(open(f"data/processed/splits/split_{dataset}/test.pkl", 'rb')))]).intersection(set([item for sublist in miivg.values() for item in sublist]))
    is2 = set([int(p) for p in patient_data.keys()]).intersection(set([item for sublist in miivg.values() for item in sublist]))
    is1.intersection(is2)
    len(is1), len(is2)
    with open(test_ids_path, 'rb') as f:
        test_ids = set(pickle.load(f))
        if dataset == 'miiv': # should be resolved better in step4 TODO 
            test_ids_path = f"data/processed/splits/split_{dataset}_demo/test.pkl"
    # Filter patients for test mode and apply threshold
    # intersection patient_data.keys() and miiv_grouped.keys()

    patients = [p for p in patient_data.keys() if int(p) in test_ids]
    if len(patients) > threshold:
        patients = patients[:threshold]

    print(f"Processing {len(patients)} patients in '{dataset}' dataset")
    patient_stats = {}

    # Process each patient
    for patient in tqdm(patients):
        patient_stats[patient] = {}
        for feature_key, data_df in patient_data[patient].items():
            feature_name = feature_key.replace(f"{dataset}_", "")

            if feature_name.startswith('static'):
                if data_df.shape[0] == 1:
                    values = data_df.iloc[0].values
                    for col_name, value in zip(data_df.columns, values):
                        patient_stats[patient][col_name] = value
                else:
                    patient_stats[patient][feature_name] = np.nan
            elif feature_name.startswith('ts'):
                feature_series = data_df.iloc[:, 0]
                res = split_timeseries_windows(feature_series)
                for moment_name, value in res.items():
                    patient_stats[patient][f"{feature_name}_{moment_name}"] = value

        # Fill missing features with NaNs
        keys_before_loop = list(patient_stats[patient].keys())
        for feature in ALL_FEATURES:
            feature_base = '_'.join(feature.split('_')[1:])
            if not any(k.startswith(feature_base) for k in keys_before_loop):
                for window_name in window_names:
                    for func in functions:
                        patient_stats[patient][f"{feature_base}_{window_name}_{func}"] = np.nan

    # Convert to DataFrame, impute missing values, and save results
    adf = pd.DataFrame.from_dict(patient_stats).T
    adf = adf.applymap(lambda x: x if isinstance(x, float) else np.nan)
    adf_without_nan_col = adf.dropna(axis=1, how='all')
    imputer = KNNImputer(n_neighbors=1, weights="uniform", missing_values=np.nan)
    imputed_values = imputer.fit_transform(adf_without_nan_col.values)
    output_folder = f"data/embeddings/stats_{dataset}"
    os.makedirs(output_folder, exist_ok=True)
    try:
        imputed_df = pd.DataFrame(imputed_values, index=adf_without_nan_col.index, columns=adf_without_nan_col.columns)
    except: 
        import pdb; pdb.set_trace()
    
    output_csv_path = os.path.join(output_folder, f"stats_{mode}_{dataset}_patient_embeddings.csv")
    print(f"Saving patient embeddings to '{output_csv_path}'")
    imputed_df.to_csv(output_csv_path)

if __name__ == "__main__":
    main()
