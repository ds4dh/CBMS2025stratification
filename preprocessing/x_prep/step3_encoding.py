import numpy as np 
import os
import pickle
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import numpy as np
import random
import torch
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
from utils.config import PRIMARY_KEY_MAP, ALL_FEATURES, TIME_VARS_MAP_DATASET_LIST_KEYS, categorical_features, DATASETS
timestart = pd.Timestamp.now()
STRING_FEATURES = list(categorical_features.keys())
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='demo', help='datasets to use')
parser.add_argument('--INPUT_BASE_DIR', type=str, default=None, help='input directory')
parser.add_argument('--OUTPUT_BASE_DIR', type=str, default=None, help='input directory')
# cut -1 24 48 or 72 
parser.add_argument('--cut', type=int, default=24*5, help='cut time by x hours 5 days')
args = parser.parse_args()
datasets = args.datasets
if datasets =='demo':  
    DATASETS = ('mimic_demo',
                # 'eicu_demo'
                )
else: 
    DATASETS = ( 'miiv', 'sic', 'eicu') 
    # DATASETS = ( 'mimic') 

INPUT_BASE_DIR = args.INPUT_BASE_DIR
OUTPUT_BASE_DIR = args.OUTPUT_BASE_DIR 
if INPUT_BASE_DIR is None:
    INPUT_BASE_DIR = os.path.join("data", "processed")
if OUTPUT_BASE_DIR is None:
    OUTPUT_BASE_DIR = os.path.join("data", "processed")

ENCODERS_DIR = 'data/encoders'
print(f'Processing datasets: {DATASETS}, input base dir: {INPUT_BASE_DIR}, output base dir: {OUTPUT_BASE_DIR} encoders dir: {ENCODERS_DIR}')
os.makedirs(ENCODERS_DIR, exist_ok=True)


def rename_pk_tk(df,dataset):
    df.rename(columns={PRIMARY_KEY_MAP[dataset]: 'pkey'}, inplace=True)
    tkeys = TIME_VARS_MAP_DATASET_LIST_KEYS[dataset]
    tkey_used = [col for col in df.columns if col in tkeys]
    if len(tkey_used) > 0:
        df.rename(columns={tkey_used[0]: 'tkey'}, inplace=True)
    # drop rows where pkey is na    
    df.dropna(subset=['pkey'], inplace=True) 
    # df['pkey'] = (pivot + df['pkey'].astype(str)).astype(int) # needed for fusion TODO
    df['pkey'] = (df['pkey'].astype(str)).astype(int)
    return df


def encode(dfall, dftemp, encoding_type, string_feature):
    encoding_type = categorical_features[string_feature]
    if encoding_type == 'one_hot': # only adm 
        encoder=OneHotEncoder(sparse_output=False)
        encoder.fit(dfall[[string_feature]])
        try:
            encoded_array = encoder.transform(dftemp[[string_feature]]) # .toarray()  # TODO check fit and transform ?
        except:
            import pdb; pdb.set_trace()
        encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out([string_feature]))
        encoded_df.index = dftemp.index
        dftemp = pd.concat([dftemp.drop(columns=[string_feature]), encoded_df], axis=1)
        if string_feature in dftemp.columns:
            dftemp.drop(columns=[string_feature], inplace=True)
        print(f'One-hot encoded {string_feature}')
    elif type(encoding_type)==list:
        ordinal_list = encoding_type
        encoder = OrdinalEncoder(categories=[ordinal_list])
        encoder.fit(dfall[[string_feature]])
        dftemp[string_feature] = encoder.transform(dftemp[[string_feature]])
        print(f'Ordinal encoded {string_feature}')
    elif encoding_type == 'boolean':
        encoder = OrdinalEncoder() # same as one hot with less space for boolean 
        encoder.fit(dfall[[string_feature]])
        dftemp[string_feature] = encoder.transform(dftemp[[string_feature]])
    else:
        raise ValueError(
            f"Unknown encoding type ''{encoding_type}'' not in {set(categorical_features.keys())} please implement consider adding in config")
    assert type(dftemp) == pd.DataFrame, f"Expected DataFrame, got {type(dftemp)}"
    encoder_path = os.path.join(ENCODERS_DIR, f'{string_feature}_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        print(f'Saving encoder to {encoder_path}')
        pickle.dump(encoder, f)
    return dftemp, encoder


def main():
    for feature in tqdm(ALL_FEATURES, desc="Processing features"):
        dfall = []
        for dataset in DATASETS:
            # import pdb; pdb.set_trace()
            input_folder = os.path.join(INPUT_BASE_DIR,f'step_2_{dataset}')
            feature_filename = f'{dataset}_{feature}.csv'
            feature_path = os.path.join(input_folder, feature_filename)
            assert os.path.exists(input_folder), f"Expected {input_folder} to exist"
            if not os.path.exists(feature_path):
                print(f"Feature {feature} not found in dataset {dataset}. Skipping.")
                continue
            df = pd.read_csv(feature_path)
            df = rename_pk_tk(df,dataset)
            dfall.append(df)

        # We aggregate all the dfs of the same feature across datasets to have uniformized encodings across models for the same values.
        if len(dfall) == 0:
            print(f"Feature {feature} not found in any dataset. Skipping.")
            continue

        dfall = pd.concat(dfall)
        for dataset in DATASETS:
            output_folder = os.path.join(INPUT_BASE_DIR,f'step_3_{dataset}')
            os.makedirs(output_folder, exist_ok=True)
            feature_path = os.path.join(INPUT_BASE_DIR,f'step_2_{dataset}', f'{dataset}_{feature}.csv')
            if not os.path.exists(feature_path):
                print(f"Feature {feature} not found in dataset {dataset}. Skipping.")
                continue
            dftemp = pd.read_csv(feature_path)
            string_feature = '_'.join(feature.split('_')[1:])
            # encode string features
            if any(feature_sub in feature for feature_sub in STRING_FEATURES):
                try:
                    encoding_type = categorical_features[string_feature]
                    dftemp, encoder = encode(dfall, dftemp, encoding_type, string_feature) 
                except:
                    print(f"Feature {string_feature} could not be found in dataset {dataset}. Skipping.")
                    continue
            is_ts_flag = 'ts_' in feature
            if is_ts_flag:
                all_tkey = TIME_VARS_MAP_DATASET_LIST_KEYS[dataset]
                # tkey in all_tkey
                tkey = [col for col in dftemp.columns if col in all_tkey][0]
                dftemp = dftemp[dftemp[tkey] <= args.cut]

            feature_filename = f'{dataset}_{feature}.csv'
            feature_output_path = os.path.join(output_folder, feature_filename)
            print('Saving feature to path:', feature_output_path)

            dftemp.to_csv(feature_output_path, index=False)
            assert len(os.listdir(output_folder)) < 130, f"Expected less than 130 files in {DATASETS[0]}"

    timeend = pd.Timestamp.now()
    print(f"Encoding completed in {timeend - timestart}")

if __name__ == '__main__':
    main()