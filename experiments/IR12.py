# from cuml.manifold import TSNE
# from cuml.cluster import KMeans
# from cuml.metrics.pairwise_distances import pairwise_distances
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
from tqdm import tqdm 
import shelve
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils.config import PRIMARY_KEY_MAP, LABEL_KEY_MAP, fpaths_lstm
from sklearn.metrics import (
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
    silhouette_score,)
import numpy as np
import pandas as pd
print('Datetime:', pd.Timestamp.now())
import argparse
from sklearn.preprocessing import LabelEncoder
import json
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances

import optuna
import pickle
from eval.metrics import compute_clustering_metrics, compute_descriptive_statistics, compute_IR_metrics
from eval.get_xy import get_data

# Define a configuration class or load from a config file
import matplotlib.pyplot as plt
import seaborn as sns

class Config:
    def __init__(self, config_file=os.path.join('utils','config.json')):
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        self.cluster_results_dir = 'data/clustering_results_dir'
        print('output_dir:', self.cluster_results_dir)
        os.makedirs(self.cluster_results_dir, exist_ok=True)
        for k, v in cfg.items():
            if k != 'data/clustering_results_dir':
                print('setting attribute:', k, v)
                setattr(self, k, v)

config = Config()

class ClusteringPipeline:
    def __init__(self, dataset, trial_count, fpath, pk, lk, mode, config, optimize):
        self.dataset = dataset
        self.trial_count = trial_count
        self.fpath = fpath
        self.pk = pk
        self.lk = lk
        self.mode = mode
        self.config = config
        self.optimize = optimize
        self.all_results = []
        self.all_ir_results = []
        self.all_descriptive_stats = []
        self.best_params_dict = {}
        self.load_data()
        
    def load_data(self):
        # Load training data
        (self.xy_filtered_train, 
         self.top75, 
         self.icd2icd9, 
         self.y_f_train, 
         self.x_f_train, 
         self.y_FL1_train, 
         self.y_FL2_train, 
         self.y_FL3_train, 
         self.y_FCCS_train, 
         self.y_FL_ICD9_train) = get_data(
             self.dataset, self.pk, self.lk, 'val', self.mode, self.fpath, 'val') 
        #  self.dataset, self.pk, self.lk, 'val', self.mode, self.fpath, 'val') 
        # use validation set for trainign the clustering model - train was used for SSL training

    def hparam(self):
        label_levels_train = [
            ('CCS', self.x_f_train, self.y_f_train),
            ('L1', self.x_f_train, self.y_FL1_train),
            ('L2', self.x_f_train, self.y_FL2_train),
            ('L3', self.x_f_train, self.y_FL3_train),
            ('icd9', self.x_f_train, self.y_FL_ICD9_train),
        ]
        os.makedirs(self.config.cluster_results_dir, exist_ok=True)
        for level_name, x_data, y_data in label_levels_train:
            print(f'Optimizing clustering at level: {level_name}')
            fpath = os.path.join(self.config.cluster_results_dir, f'{self.mode}_{self.dataset}_optuna_study_tc_{str(self.trial_count)}_{level_name}.pkl')
            print('fpath:', fpath, 'is used for data')
            if os.path.exists(fpath):

                if self.optimize: # overide option to load if --optimize is passed
                    best_params = self.train_study(x_data, y_data, level_name)
                else:
                    with open(fpath, 'rb') as f:
                        print(f"Loading study from {fpath}")
                        study = pickle.load(f)
                        best_params = study.best_trial.params
            else:
                best_params = self.train_study(x_data, y_data, level_name)
            self.best_params_dict[level_name] = best_params
        # Save best_params_dict
        self.json_fpath = os.path.join(self.config.cluster_results_dir, f'{self.mode}_{self.dataset}_best_params_dict_tc_{str(self.trial_count)}_{level_name}.json')
        # import pdb; pdb.set_trace()
        with open(self.json_fpath, 'w') as f:
            print(f"################################# Saving best parameters dict for all pipeline to {self.json_fpath}")
            json.dump(self.best_params_dict, f)
        return self.best_params_dict

    def train_study(self, x_data, y_data, level_name):
        study = optuna.create_study(direction=self.config.optuna_direction, study_name=level_name)
        study.optimize(lambda trial: self.objective(trial, x_data, y_data, level_name), n_trials=self.trial_count)
        fpath = os.path.join(self.config.cluster_results_dir, f'{self.mode}_{self.dataset}_optuna_study_tc_{str(self.trial_count)}_{level_name}.pkl')
        with open(fpath, 'wb') as f:
            print(f"Saving study to {fpath}")
            pickle.dump(study, f)
            print(f"Best parameters for {level_name}: {study.best_trial.params}")
        best_params = study.best_trial.params
        return best_params

    def objective(self, trial, x, y, group_name):
        if len(y) < 2 or len(x) < 4:
            print(f'Not enough samples for {group_name} with y={len(y)}')
            return 0  # Return a minimal score

        # Sample t-SNE parameters within reasonable bounds
        perplexity = trial.suggest_int('perplexity', 
                                       min(5, len(x)-1), 
                                       min(50, len(x)-1))
        learning_rate = trial.suggest_float('learning_rate', 10.0, 1000.0, log=True)
        # n_iter = 300        # Sample number of clusters
        n_iter =  trial.suggest_int('n_iter', 250, 251)
        # n_iter =  trial.suggest_int('n_iter', 250, 2000)

        y_unique = len(set(y))
        num_clusters = trial.suggest_int('num_clusters', 2, 
                                         min(self.config.max_num_clusters, y_unique*10))
       

        y_encoded, y_pred, kmeans = self.cluster(num_clusters, x, y, perplexity, learning_rate, n_iter)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_encoded, y_pred)
        ami = adjusted_mutual_info_score(y_encoded, y_pred)
        return v_measure

    def determine_best_params(self, y, group, best_params=None):
        num_samples = len(y)
        if len(set(y)) < 2:
            print(f'Not enough unique labels for {group} with y={len(y)}')
            return None, None, None, None
        if num_samples < self.config.min_samples_per_group:
            print(f'Not enough samples for {group} with y={len(y)}')
            return None, None, None, None
        if best_params is None:
            print('best_params is None for determine_number_of_clusters of, group:', group)
            unique_labels = len(set(y))
            num_clusters = min(self.config.max_num_clusters, unique_labels)
            if num_clusters < 2:
                print(f'1 cluster for {group} with y={len(y)}')
                return None, None, None, None
            perplexity = min(self.config.default_perplexity, unique_labels)
            learning_rate = self.config.default_learning_rate
            self.config.default_n_iter
        else:
            num_clusters = best_params['num_clusters']
            perplexity = best_params['perplexity']
            learning_rate = best_params['learning_rate']
            n_iter = best_params['n_iter']
        print(f'Group: {group} - Number of clusters: {num_clusters}')
        if perplexity >= num_samples:
            perplexity = num_samples - 1
        num_clusters = min(num_clusters, len(set(y)))
        return num_clusters, perplexity, learning_rate, n_iter

    def cluster(self, num_clusters, x, y, perplexity, learning_rate, n_iter):
        num_samples = len(y)
        print('Number of samples for clustering:', num_samples)
        print('Perplexity:', perplexity)

        tsne = TSNE(
            n_components=min(self.config.tsne_n_components, num_samples),
            random_state=self.config.tsne_random_state,
            perplexity=min(self.config.default_perplexity, len(y)-1),
            n_iter=self.config.default_n_iter,
            learning_rate=self.config.learning_rate,
            verbose=1,
        )
        try:          
            x_transformed = tsne.fit_transform(x)   
        except:
            import pdb; pdb.set_trace()
        if num_clusters == None:
            num_clusters=min(self.config.max_num_clusters, len(y)-1)
        print('x_transformed shape:', x.shape)
        model = KMeans(
            n_clusters=min(num_clusters, len(x_transformed)),
            random_state=self.config.kmeans_random_state
        ).fit(x_transformed)
        y_pred = model.predict(x_transformed)
        y_encoded = LabelEncoder().fit_transform(y)
      

        return y_encoded, y_pred, model

    def cluster_model(self, x_train, x_test, y_true_train, y_true_test, group, best_params=None, descr_stats=False):
        '''
        Cluster the data and evaluate the clustering metrics, 

        '''
        print('x train shape:', len(x_train))
        print('y train shape:', len(y_true_train))
        print('x test shape:', len(x_test))
        print('y test shape:', len(y_true_test))
        print('Clustering for group:', group)
        
        try: 
            num_clusters, perplexity, learning_rate, n_iter = self.determine_best_params(y_true_train, group, best_params)
        except Exception as e:
            import pdb; pdb.set_trace()
            print(f"Error in cluster_model: {e}")
            return 
        if num_clusters is None:
            # import pdb; pdb.set_trace()
            number_of_clusters = 10
        y_encoded, y_pred_test, model = self.cluster(num_clusters, x_test, y_true_test, perplexity, learning_rate, n_iter)
        self.model = model
        print('DONE cluster y_encoded shape:', len(y_encoded))
        
        try:
            print('compute_clustering_metrics')
            self.all_results = compute_clustering_metrics(self.all_results, y_encoded, y_pred_test, x_test, group, 'KMeans')
        except Exception as e:
            print(f"Error in cluster_model: {e}")
            print([type(x) for x in [x_train, x_test, y_true_train, y_true_test, group, best_params]])
            print([len(x) for x in [x_train, x_test, y_true_train, y_true_test]])
            print(x_train)
            import pdb; pdb.set_trace()

        try:
        
            self.all_ir_results = compute_IR_metrics(self.all_ir_results,x_train,x_test,y_true_train,y_true_test,model=self.model,group=group,method='KMeans',dataset=self.dataset,results_dir=self.config.cluster_results_dir,)
            print('\n\n\n ### all_results length:', len(self.all_results))
        except Exception as e:
            print(f"Error in cluster_model: {e}")
            print([type(x) for x in [x_train, x_test, y_true_train, y_true_test, group, best_params]])
            print([len(x) for x in [x_train, x_test, y_true_train, y_true_test]])
            print(x_train)
            import pdb; pdb.set_trace()
        
        try:
        
            if descr_stats:
                self.train_stats = compute_descriptive_statistics(x_train, y_true_train, group)
                self.test_stats = compute_descriptive_statistics(x_test, y_true_test, group)
            print('### IR metrics for dataset:')
            print(self.dataset)

        except Exception as e:
            print(f"Error in cluster_model: {e}")
            print([type(x) for x in [x_train, x_test, y_true_train, y_true_test, group, best_params]])
            print([len(x) for x in [x_train, x_test, y_true_train, y_true_test]])
            print(x_train)
            import pdb; pdb.set_trace()
        
        def save_descriptive_stats(stats, dataset, group, output_dir, split_name):
            """
            Save descriptive statistics for train or test datasets in both JSON and CSV formats.

            Args:
                stats (dict): Descriptive statistics dictionary.
                dataset (str): Name of the dataset.
                group (str): Group identifier.
                output_dir (str): Directory where the files will be saved.
                split_name (str): Split type ('train' or 'test').
            """
            # File paths
            json_path = os.path.join(output_dir, f'{dataset}_{split_name}_stats_{group}.json')
            x_stats_csv_path = os.path.join(output_dir, f'{dataset}_{split_name}_x_stats_{group}.csv')
            y_counts_csv_path = os.path.join(output_dir, f'{dataset}_{split_name}_y_counts_{group}.csv')

            # Save x_stats as CSV
            stats['x_stats'].to_csv(x_stats_csv_path, index=True)
            print(f"Saved x_stats to {x_stats_csv_path}")

            # Save y_value_counts as CSV
            stats['y_value_counts'].to_csv(y_counts_csv_path, index=True)
            print(f"Saved y_value_counts to {y_counts_csv_path}")

            # Save entire stats dictionary as JSON
            with open(json_path, 'w') as f:
                json.dump({
                    'group': stats['group'],
                    'x_stats': stats['x_stats'].to_dict(),
                    'y_value_counts': stats['y_value_counts'].to_dict()
                }, f, indent=4)
            print(f"Saved complete stats to {json_path}")

            # Assert tests
            assert os.path.exists(x_stats_csv_path), f"x_stats CSV was not saved at {x_stats_csv_path}!"
            assert os.path.exists(y_counts_csv_path), f"y_value_counts CSV was not saved at {y_counts_csv_path}!"
            assert os.path.exists(json_path), f"Stats JSON was not saved at {json_path}!"

        
        output_dir = self.config.cluster_results_dir
        if descr_stats:
            save_descriptive_stats(self.train_stats, self.dataset, group, output_dir, split_name='train')
            save_descriptive_stats(self.test_stats, self.dataset, group, output_dir, split_name='test')
            # if group == 'icd9':
            #     import pdb; pdb.set_trace()
            #     self.all_ir_results,
            #     x_train,
            #     x_test,
            #     y_true_train,
            #     y_true_test,
            #     group=group,
            #     method='KMeans'
            #     dataset=self.dataset,
            #     kmean_test,
            # max_clusters=5 #ODO check 
        

    def cluster_within_group(self, df, group_column, target_column, exp_name, desc_stats=False):
    
        print('\n\nEvaluating clustering within groups')
        groups = df[group_column].unique()
        for group in groups:
            group_df = df[df[group_column] == group]
            group_y = group_df[target_column].values
            group_x = group_df.drop([self.pk, self.lk] + self.config.labels_to_drop, axis=1).values
            if len(group_y) < self.config.min_samples_per_group:
                print(f'Not enough samples for group {group} in {exp_name}')
                continue
            if desc_stats:
                compute_descriptive_statistics(group_x, group_y, exp_name + '_' + str(group))
            best_params = self.best_params_dict.get(target_column, None)
            if best_params is None:
                print(f"cluster_within_group NO BEST  parameters found for level {target_column}")
                continue
            for key in best_params.keys():
                if best_params[key] is None:
                    best_params[key] = config.__dict__[key]
                    print('set best_params:', key, best_params[key])
            x_train, x_test, y_train, y_test = train_test_split(
                group_x, group_y, test_size=0.50, random_state=self.config.random_state)
            if x_train.shape[0] < self.config.min_samples_per_group or x_test.shape[0] < self.config.min_samples_per_group:
                print(f'Not enough samples for group {group} in {exp_name}')
                continue
            self.cluster_model(x_train, x_test, y_train, y_test, exp_name + '_' + str(group), best_params)

    def eval_test_set(self, desc_stat=False):
        # Load test data
        (self.xy_filtered_test,
         top75,
         icd2icd9,
         self.y_f_test,
         self.x_f_test,
         self.y_FL1_test,
         self.y_FL2_test,
         self.y_FL3_test,
         self.y_FCCS_test,
         self.y_FL_ICD9_test) = get_data(
             self.dataset, self.pk, self.lk, 'test', self.mode, self.fpath, 'test')

        label_levels_test = [
            ('CCS', self.x_f_test, self.y_f_test),
            ('L1', self.x_f_test, self.y_FL1_test),
            ('L2', self.x_f_test, self.y_FL2_test),
            ('L3', self.x_f_test, self.y_FL3_test),
            ('icd9', self.x_f_test, self.y_FL_ICD9_test),
        ]

        for level_name, x_data, y_data in label_levels_test:
            print(f'Clustering at level: {level_name}')
            x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
                x_data, y_data, test_size=self.config.test_size, random_state=self.config.random_state)
            # if desc_stat: 
            # stats = compute_descriptive_statistics(x_data, y_data, level_name)
            # self.all_descriptive_stats.append(stats)
            best_params = self.best_params_dict.get(level_name)
            if best_params is None:
                print(f"No best parameters found for level {level_name}")
                # continue
            self.cluster_model(x_data_train, x_data_test, y_data_train, y_data_test, level_name, best_params)


    def eval_subgroups(self):
        output_dir = self.config.clustering_results_dir
        for group_col, target_col, exp_name in self.config.groupings:
            print(f'\n\nClustering within group: {exp_name}')
            self.cluster_within_group(self.xy_filtered_test.copy(), group_col, target_col, exp_name)
            df_metrics = pd.DataFrame(self.all_results)
            clustering_metrics_fpath = os.path.join(output_dir, f'clustering_metrics_{self.mode}_{self.dataset}.csv')
            os.makedirs(output_dir, exist_ok=True)
            df_metrics.to_csv(clustering_metrics_fpath, index=False)
            print(f"Saved clustering metrics to {clustering_metrics_fpath}")
            os.makedirs(output_dir, exist_ok=True)
            df_ir = pd.DataFrame(self.all_ir_results)
            ir_path = os.path.join(output_dir, f'clustering_ir_metrics_{self.mode}_{self.dataset}.csv')
            df_ir.to_csv(ir_path, index=False)



            output_dir = self.config.clustering_results_dir
            print('output_dir:', self.config.clustering_results_dir)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saved IR metrics to {ir_path}")
            # import pdb; pdb.set_trace()
            # with open(os.path.join(output_dir, f'all_descriptive_statistics_{self.mode}_{self.dataset}.json'), 'w') as f:
            #     json.dump([{'group': stat['group'], 'x_stats': stat['x_stats'].to_dict(), 'y_value_counts': stat['y_value_counts'].to_dict()} for stat in self.all_descriptive_stats], f)

    def run(self):
        if self.trial_count > 0:
            self.hparam()
        self.eval_test_set()
        self.eval_subgroups()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic_demo', help='Dataset to use')
    parser.add_argument('--trials', type=int, default=1, help='Number of Optuna trials')
    parser.add_argument('--mode', type=str, default='stats', choices=['stats', 'lstm', 'gru'])
    parser.add_argument('--optimize', action='store_true', help='Optimize hyperparameters')
    parser.add_argument('--fpath', type=str, required=True, help='Path to embeddings')
    args = parser.parse_args()
    mode = args.mode
    if mode == 'lstm' or mode == 'gru':
        assert args.fpath is not None, "Please provide a path to the embeddings"
    dataset = args.dataset
    trial_count = args.trials
    fpath = args.fpath
    optimize = args.optimize
    pk, lk = PRIMARY_KEY_MAP[dataset], LABEL_KEY_MAP[dataset]
    return dataset, trial_count, fpath, pk, lk, mode, optimize

def main():
    dataset, trial_count, fpath, pk, lk, mode, optimize = parse_args()
    pipeline = ClusteringPipeline(dataset, trial_count, fpath, pk, lk, mode, config,optimize)
    pipeline.run()

if __name__ == '__main__':
    main()
# python -m experiments.IR12 --dataset mimic_demo --trials 10 --optimize --mode stats --fpath data/embeddings/stats_mimic_demo/stats_test_mimic_demo_patient_embeddings.csv
# python -m experiments.IR12 --dataset mimic_demo --trials 10 --optimize --mode lstm --fpath data/embeddings/singleLSTM_train_mimic_demo_e_10_ms_10000_samples_10000__bs_2.shelve
# python -m experiments.IR12 --dataset mimic_demo --trials 10 --optimize --mode gru --fpath data/embeddings/gru_train_mimic_demo_e_10_ms_10000_samples_10000__bs_2.shelve

# python -m experiments.IR12 --dataset eicu --trials 0 --mode stats --fpath data/cluster_results/eicu_stats_embeddings.pkl > data/cluster_results/eicu_stats.log 2>&1
# os0@gpu035 ricu_clean]$ python -m experiments.IR12 --dataset eicu --trials 5 --optimize --mode gru --fpath data/embeddings/eicu_gru_embeddings_eicu_ms100k
# config imported 
# python -m experiments.IR12 --dataset eicu --trials 10 --optimize --mode gru --fpath data/embeddings/eicu_gru_embeddings_eicu_ms100k
# python -m experiments.IR12 --dataset eicu --trials 10 --optimize --mode lstm --fpath data/embeddings/eicu_lstm_embeddings_eicu_ms100k
# python -m experiments.IR12 --dataset sic --trials 10 --optimize --mode lstm --fpath data/embeddings/sic_egru_mbeddings_sic_ms100k

# apptainer pull rapids-23.06.sif docker://rapidsai/rapidsai:23.06-cuda11.8-runtime-ubuntu22.04-py3.10
# apptainer shell --nv rapids-23.06.sif
# python -m experiments.IR12 --dataset eicu --trials 1 --optimize --mode stats  --fpath data/embeddings/stats_eicu/stats_test_eicu_patient_embeddings.csv


# Apptainer> source activate cuml
# apptainer pull rapids-23.08a-cuda12.0.1.sif docker://rapidsai/rapidsai:23.08a-cuda12.0.1-py3.10
# apptainer shell --nv rapids-23.08a-cuda12.0.1.sif
# apptainer shell --nv --contain cuml.sif
# 

# apptainer pull docker://rapidsai/rapidsai:23.08a-cuda12.0.1-py3.10