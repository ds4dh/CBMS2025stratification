# PatientDataset.py

import glob
import pickle
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.config import ALL_FEATURES, get_feature_name_from_file

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_patient_ids(pkl_file):
    """Helper function to load patient ids from a pickle file."""
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

def create_dataloaders(
    max_patients: int,
    batch_size: int,
    dataset: str = 'mimic_demo',
    data_pickle: str = None,
    DEBUG_FLAG=False,
    vcut: int = 1500
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load subset of data from a pickle file and create training, validation, and testing dataloaders."""

    if data_pickle is None:
        data_pickle = f'data/processed/step_5_{dataset}/{dataset}_processed_data.pkl'

    base_dir = f'data/processed/splits/split_{dataset}'
    
    # Load patient ids for each data split
    train_ids = load_patient_ids(f'{base_dir}/train.pkl')
    val_ids = load_patient_ids(f'{base_dir}/val.pkl')
    test_ids = load_patient_ids(f'{base_dir}/test.pkl')

    if DEBUG_FLAG:
        logging.info("DEBUG MODE: Using only 10 patients for each split")
        train_ids, val_ids, test_ids = train_ids[:10], val_ids[:10], test_ids[:10]

    logging.info(f"Loaded {len(train_ids)} training, {len(val_ids)} validation, and {len(test_ids)} testing patient IDs")

    # Load the pickle file ONCE and pass the data instead of the file path
    logging.info(f"Loading data from pickle: {data_pickle}")
    with open(data_pickle, 'rb') as f:
        patient_data = pickle.load(f)

    # Build datasets with preloaded data
    train_ds = PatientDataset(
        dataset_name=dataset,
        patient_ids=train_ids,
        patient_data=patient_data,
        max_patients=max_patients,
        mode_dataset='train',
        vcut=vcut  # Pass vcut here
    )
    val_ds = PatientDataset(
        dataset_name=dataset,
        patient_ids=val_ids,
        patient_data=patient_data,
        max_patients=max_patients,
        mode_dataset='val',
        vcut=vcut
    )
    test_ds = PatientDataset(
        dataset_name=dataset,
        patient_ids=test_ids,
        patient_data=patient_data,
        max_patients=max_patients,
        mode_dataset='test',
        vcut=vcut
    )

    logging.info(f"Number of patients with data in training set: {len(train_ds)}")
    logging.info(f"Number of patients with data in validation set: {len(val_ds)}")
    logging.info(f"Number of patients with data in testing set: {len(test_ds)}")

    # Determine number of workers
    num_workers = 12 if not DEBUG_FLAG else 0

    # Create DataLoaders
    train_dl = SafeDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    val_dl = SafeDataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    test_dl = SafeDataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )

    return train_dl, val_dl, test_dl

class PatientDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        patient_ids,
        patient_data,
        feature_mins=None,
        feature_maxs=None,
        feature_means=None,
        feature_stds=None,
        normalize_flag=False,
        norm_mode="min-max",
        max_patients=-1,
        mode_dataset=None,
        vcut=None  # Added vcut parameter
    ):
        if mode_dataset is None:
            raise ValueError("mode_dataset must be specified")

        self.dataset_name = dataset_name
        self.normalize_flag = normalize_flag
        self.norm_mode = norm_mode
        self.patient_data = patient_data  # Directly use loaded patient data
        self.vcut = vcut  # Store vcut

        # Limit patients
        if max_patients > 0:
            patient_ids = patient_ids[:max_patients]

        # Filter patient IDs
        self.patient_ids = [
            pid for pid in tqdm(patient_ids, desc=f"For {mode_dataset} filtering patients with data")
            if self.check_patient_has_data(pid)
        ]

        logging.info(f"Timestamp: {pd.Timestamp.now()}")
        self.num_patients_with_data = len(self.patient_ids)
        self.num_patients_without_data = len(patient_ids) - self.num_patients_with_data

        # Feature statistics for normalization
        self.feature_mins = feature_mins
        self.feature_maxs = feature_maxs
        self.feature_means = feature_means
        self.feature_stds = feature_stds

    def check_patient_has_data(self, patient_id):
        return patient_id in self.patient_data and len(self.patient_data[patient_id]) > 1

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        """Retrieve a single data sample, skipping empty patients."""
        try:
            patient_id = self.patient_ids[idx]
            raw_features = self.gather_features(patient_id)

            # Validate data
            if raw_features is None or raw_features.empty or raw_features.shape[0] <= 1:
                raise ValueError(f"Patient {patient_id} has insufficient data.")

            # Normalize features
            features = self._normalize_features(raw_features)
            feature_tensor = torch.tensor(features.values, dtype=torch.float32)

            # Create input and target sequences
            input_seq = feature_tensor[:-1]
            target_seq = feature_tensor[1:]
        except Exception as e:
            logging.warning(f"Skipping patient at index {idx} due to error: {e}")
            import pdb; pdb.set_trace()
            # Return None to indicate a bad sample
            return None

        try:

            self.vcut = min(self.vcut, input_seq.size(0)) if self.vcut is not None else None
            # Enforce vcut if specified
            if self.vcut is not None:
                if input_seq.size(0) > self.vcut:
                    input_seq = input_seq[:self.vcut]
                    target_seq = target_seq[:self.vcut]

            return patient_id, input_seq, target_seq

        except Exception as e:
            logging.warning(f"Skipping patient at index {idx} due to error: {e}")
            import pdb; pdb.set_trace()
            # Return None to indicate a bad sample
            return None

    def gather_features(self, patient_id) -> pd.DataFrame:
        """Gather and preprocess features for a single patient."""

        if patient_id not in self.patient_data:
            final_features = pd.DataFrame()
        else:
            patient_features = self.patient_data[patient_id]
            results = {}
            map_feature_type = {}

            # Gather features from the pickle data
            for feature_key, data_df in patient_features.items():
                feature_name = feature_key.replace(f"{self.dataset_name}_", "")
                results[feature_name] = data_df
                if feature_name.startswith('static'):
                    map_feature_type[feature_name] = 'static'
                elif feature_name.startswith('ts'):
                    map_feature_type[feature_name] = 'ts'

            # Handle missing features by assigning -100
            for feature_name in ALL_FEATURES:
                feature_base = feature_name
                if feature_base not in results:
                    if feature_name.startswith('static_'):
                        # Assign -100 or a list of -100s based on feature specifics
                        if 'adm' in feature_base:
                            results[feature_base] = pd.DataFrame(
                                {feature_base: [[-100, -100, -100]]},
                                columns=['adm_med', 'adm_other', 'adm_surg'],
                                dtype=float,
                            )
                        else:
                            results[feature_base] = pd.DataFrame({feature_base: [-100]})
                        map_feature_type[feature_base] = 'static'
                    elif feature_name.startswith('ts_'):
                        results[feature_base] = pd.DataFrame({'time': -1, feature_base: [-100]})
                        map_feature_type[feature_base] = 'ts'

            # Ensure total number of features is within expected bounds
            assert len(results) < 130, f"Number of features for patient {patient_id} exceeds 130: {len(results)}"

            # Separate static and time-series features
            static_features = [key for key in results if map_feature_type.get(key) == 'static']
            ts_features = [key for key in results if map_feature_type.get(key) == 'ts']

            # Determine the maximum timestep across all ts features
            max_timestep = max([results[key].shape[0] for key in ts_features], default=1)

            # Process static features: forward fill to match max_timestep
            static_padded = []
            for key in static_features:
                df = results[key].reindex(range(max_timestep), method='ffill').fillna(-100)
                static_padded.append(df)
            concat_static = pd.concat(static_padded, axis=1) if static_padded else pd.DataFrame()

            # Process time-series features: merge with a time grid and forward fill
            merged_ts = []
            for key in ts_features:
                df = results[key].copy()
                time_colname = df.columns[0]
                df = df.sort_values(by=time_colname).reset_index(drop=True)
                df = df.reindex(range(max_timestep), method='ffill').fillna(-100)
                merged_ts.append(df.drop(columns=[time_colname]))
            concat_ts = pd.concat(merged_ts, axis=1) if merged_ts else pd.DataFrame()

            # Concatenate static and time-series features
            if not concat_static.empty and not concat_ts.empty:
                final_features = pd.concat([concat_static, concat_ts], axis=1)
            elif not concat_static.empty:
                final_features = concat_static
            elif not concat_ts.empty:
                final_features = concat_ts
            else:
                final_features = pd.DataFrame()

        return final_features

    def _normalize_features(self, df, mode="min-max"):
        """Normalize features based on the selected mode:
            - "min-max": Apply Min-Max normalization: (X - min) / (max - min)
            - "z-score": Apply Z-score normalization: (X - mean) / std
            Missing (-100) values are not normalized and remain unchanged
        """
        if not self.normalize_flag:
            return df

        assert self.norm_mode in ["z-score", "min-max"], "Invalid normalization mode"

        # Mask to preserve missing values
        missing_mask = df == -100

        if self.norm_mode == "min-max":
            df_normalized = (df - self.feature_mins) / (self.feature_maxs - self.feature_mins)
            df_normalized = df_normalized.clip(0, 1)  # Ensure values are within [0, 1]
        elif self.norm_mode == "z-score":
            df_normalized = (df - self.feature_means) / self.feature_stds

        # Reinsert missing values
        df_normalized[missing_mask] = -100

        return df_normalized

    def compute_feature_statistics(self, max_samples=None):
        """Compute feature statistics for normalization."""
        all_features = []
        for i, patient_id in enumerate(tqdm(
            self.patient_ids,
            desc="Computing dataset statistics",
            total=len(self.patient_ids) if max_samples is None else max_samples,
        )):
            if max_samples is not None and i >= max_samples:
                break
            raw_features = self.gather_features(patient_id)
            if raw_features is None or raw_features.empty:
                continue
            features = raw_features.replace(-100, np.nan)
            all_features.append(features)
        if not all_features:
            return None

        all_features_concat = pd.concat(all_features, axis=0)

        # Calculate min, max, mean, std ignoring NaNs
        feature_mins = all_features_concat.min(axis=0)
        feature_maxs = all_features_concat.max(axis=0)
        feature_means = all_features_concat.mean(axis=0)
        feature_stds = all_features_concat.std(axis=0)

        # Handle features where min == max
        same_min_max = feature_mins == feature_maxs
        feature_mins[same_min_max] = 0.0
        feature_maxs[same_min_max] = 1.0

        # Fill NaNs for features entirely missing
        feature_mins.fillna(0.0, inplace=True)
        feature_maxs.fillna(1.0, inplace=True)
        feature_means.fillna(0.0, inplace=True)
        feature_stds.fillna(1.0, inplace=True)

        self.feature_mins = feature_mins
        self.feature_maxs = feature_maxs
        self.feature_means = feature_means
        self.feature_stds = feature_stds

        return feature_mins, feature_maxs, feature_means, feature_stds

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences and skip bad samples."""
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        logging.warning("Received empty batch after filtering. Returning None.")
        return None, None, None, None
    try:
        # Extract data and length of each sample sequence
        patient_id, inputs, targets = zip(*batch)
        lengths = torch.tensor([seq.size(0) for seq in inputs])

        # Ensure all sequences have the same feature dimension
        feature_dims = [seq.size(1) for seq in inputs]
        if len(set(feature_dims)) > 1:
            raise ValueError(f"Inconsistent feature dimensions in batch: {feature_dims}")

        # Pad inputs and targets to max length
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=-111).contiguous()
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=-111).contiguous()

        return patient_id, inputs_padded, targets_padded, lengths
    except Exception as e:
        logging.error(f"Error in collate_fn: {e}. Returning None for this batch.")
        return None, None, None, None

from torch.utils.data import DataLoader

class SafeDataLoader(DataLoader):
    """A DataLoader that gracefully skips batches that cause errors."""

    def __iter__(self):
        iterator = super().__iter__()
        for batch in iterator:
            try:
                yield batch
            except Exception as e:
                logging.error(f"Skipping a batch due to error: {e}")
                continue
