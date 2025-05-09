import tensorboard
import gc 
import shelve
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os
import pickle
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from models.LSTM_singleChannel import LSTM_singleChannel
from models.ChanellWiseLSTMv1 import ChannelWiseLSTM
# from models.ChanellWiseLSTM import FlexibleChannelLSTM
from models.PatientDataset1 import create_dataloaders
from utils.config import log_time

random.seed(42), np.random.seed(42), torch.manual_seed(42)

# Configure pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
os.makedirs('all_tb_logs', exist_ok=True)
os.makedirs('data/model_weights', exist_ok=True)


class MemoryCleanupCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gc.collect()
        torch.cuda.empty_cache()
        
@log_time
def save_test_embeddings(model, test_dl, output_path, dataset, mode) -> None:
    """ Save embeddings from the testing dataset using shelve for incremental saving """
    with torch.no_grad():
        model.eval()
        counter = 0
        print('Saving embeddings in:', output_path)
        print('Starting test at pd timestamp:', pd.Timestamp.now())

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Use shelve for incremental saving with a compatible file extension
        with shelve.open(output_path, writeback=False) as db:
            for batch in tqdm(test_dl, desc="Generating Embeddings"):
                counter += 1
                patient_id, inputs, targets, lengths = batch
                if inputs is None:
                    continue
                with torch.no_grad():  # Ensure no gradients are computed
                    outputs, embeddings = model(inputs, lengths)
                    # Convert embeddings to float32 to save memory
                    embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
                
                # Map each patient_id to its embedding
                for pid, emb in zip(patient_id, embeddings_np):
                    db[str(pid)] = emb  # Shelve requires string keys

                # Clear variables to free memory
                del embeddings, embeddings_np
                torch.cuda.empty_cache()
                gc.collect()

                if mode == 'debug' and counter > 10:
                    print("Debug mode: Breaking after 10 batches")
                    break

        print(f"Embeddings successfully saved to {output_path}")


@log_time
def main(gru,batch_size, data_pickle, mode, max_steps, max_epochs, dataset, max_patients, learning_rate, timeseries_model: str, mini: bool) -> None:
    train_dl, val_dl, test_dl = create_dataloaders(max_patients, batch_size, dataset, data_pickle, mode=='debug')

    # Access datasets to get counts
    train_ds = train_dl.dataset
    val_ds = val_dl.dataset
    test_ds = test_dl.dataset

    print(f"Training set: {train_ds.num_patients_with_data} patients with data, {train_ds.num_patients_without_data} without data.")
    print(f"Validation set: {val_ds.num_patients_with_data} patients with data, {val_ds.num_patients_without_data} without data.")
    print(f"Test set: {test_ds.num_patients_with_data} patients with data, {test_ds.num_patients_without_data} without data.")

    # Get input dimension from the first batch
    first_batch = next(iter(train_dl))
    num_dim = first_batch[1].shape[-1]
    print(num_dim)
    # timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    hidden_dim = 128
    print(f"define model timestamp: {pd.Timestamp.now()}")
    if timeseries_model=='singleLSTM':
        # To use single-channel mode
        model = LSTM_singleChannel(
            input_dim=num_dim,
            hidden_dim=hidden_dim,
            output_dim=num_dim,
            # num_channels=num_dim,
            lstmcell=not gru, # GRU -> gru_flag True 
            timeseries_mode='single',
            # optimizer_type='AdamW',
            learning_rate=learning_rate,
        )

    elif timeseries_model=='multiLSTM':
        # To use multi-channel mode
        model = ChannelWiseLSTM(
            input_dim=1,
            hidden_dim=hidden_dim,
            output_dim=num_dim,
            num_channels=1,
            timeseries_mode='multi',
            # optimizer_type='AdamW',
            learning_rate=learning_rate,
        )
     
    model_save_path = f'data/model_weights/{timeseries_model}_{dataset}_e_{max_epochs}_ms_{max_steps}_samples_{max_patients}__bs_{batch_size}.pt'
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{model_save_path}',  # Directory to save checkpoints
        filename='{epoch}-{val_loss:.2f}',  # Filename format
        save_top_k=-1,  # Save every epoch
        every_n_epochs=1  # Save at each epoch
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    memcbk=MemoryCleanupCallback()

    # Configure PyTorch Lightning Trainer
    print(f"define trainer  timestamp: {pd.Timestamp.now()}")
    tb_model_label = timeseries_model if not gru else 'gru'
    tbpath=f'{tb_model_label}_{mode}_{dataset}_ms_{max_steps}_mp_{max_patients}_lr_{learning_rate}_bs_{batch_size}_ep_{max_epochs}'
    if mode == 'debug':
        trainer = pl.Trainer(
            log_every_n_steps=1,
            accelerator='cpu',
            max_steps=3,
            devices=1,
            max_epochs=max_epochs,
            profiler='simple',
            logger=pl.loggers.TensorBoardLogger(
                save_dir=f'all_tb_logs/tb_logs_{dataset}',
                name=tbpath
            ),
        )
    elif mode == 'train':
        trainer = pl.Trainer(
            log_every_n_steps=10,
            # accelerator='cpu',
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            # max_steps=max_steps if max_steps > 0 else None,
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback, memcbk],  # Add checkpoint callback here
            logger=pl.loggers.TensorBoardLogger(
                save_dir=f'all_tb_logs/tb_logs_{dataset}',
                name=tbpath
            ),
        )

        print('Starting training at pd timestamp:', pd.Timestamp.now())
        trainer.fit(model, train_dl, val_dl)

    elif mode == 'test':
        trainer = pl.Trainer(
            log_every_n_steps=5,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            max_steps=max_steps,
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback],  # Add checkpoint callback here
            logger=pl.loggers.TensorBoardLogger(
                save_dir=f'all_tb_logs/tb_logs_{dataset}',
                name=f'lstm_{mode}_{dataset}_e_{max_epochs}_ms_{max_steps}_mp_{max_patients}_lr_{learning_rate}_bs_{batch_size}_ep_{max_epochs}'
            ),
        )
        print('Starting test at pd timestamp:', pd.Timestamp.now())
        trainer.test(model, test_dl)
        # Save the trained model
    else:
        raise ValueError(f"Invalid mode: {mode}")
    # save model of  best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print(f'Best checkpoint path: {best_model_path}')
    # import pdb; pdb.set_trace()
    print(f'Saving model state_dict to: {model_save_path}')
    torch.save(model.state_dict(), model_save_path)
    print('Script finished at pd timestamp:', pd.Timestamp.now())	
    save_test_embeddings(model, test_dl, f'data/embeddings/{tb_model_label}_{mode}_{dataset}_e_{max_epochs}_ms_{max_steps}_samples_{max_patients}__bs_{batch_size}.shelve', dataset, mode)


if __name__ == "__main__":
    print('Starting at pd timestamp:', pd.Timestamp.now())	
    parser = argparse.ArgumentParser(description="Train and evaluate ChannelWiseLSTM model.")
    parser.add_argument('--dataset', type=str, default='mimic_demo', help='Dataset name')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'debug', 'test'], help='Mode')
    parser.add_argument('--max_steps', type=int, default=-1, help='Max training steps')
    parser.add_argument('--max_epochs', type=int, default=10, help='Max training epochs')
    parser.add_argument('--max_patients', type=int, default=-1, help='Max number of patients to process')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate') #-3 -5
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--timeseries_model', type=str, default='singleLSTM', choices=['singleLSTM', 'multiLSTM'], help='Type of timeseries model to use')
    parser.add_argument('--mini', default=False, help='Mini dataset', action='store_true')
    parser.add_argument('--data_pickle', type=str, default=None, help='Path to the data pickle file')
    parser.add_argument('--gru', default=False, help='Use GRU instead of LSTM', action='store_true')
    args = parser.parse_args()

    dataset = args.dataset
    mode = args.mode
    max_steps = args.max_steps
    max_epochs = args.max_epochs
    max_patients = args.max_patients
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    timeseries_model = args.timeseries_model
    gru_flag = args.gru
    if args.data_pickle is None:
            args.data_pickle = f'data/processed/step_5_{args.dataset}_all/{args.dataset}_processed_data.pkl'
    if args.mini and 'demo' not in args.dataset:
        if args.max_patients == -1:
            args.max_patients = 'small'
        print(f"Using mini dataset with {args.max_patients} patients")
        args.data_pickle = f'data/processed/step_5_{args.dataset}/{args.dataset}_processed_data_small.pkl'
    assert os.path.exists(args.data_pickle), f"Data pickle file not found at: {args.data_pickle}"
    main(**vars(args))
    print('Script finished at pd timestamp:', pd.Timestamp.now())

# python -m models.lstmv5 --mode train --max_steps 1 --max_epochs 1 --max_patients 100 --learning_rate 1e-4 --batch_size 32 --timeseries_model singleLSTM --mini --dataset mimic_demo
# python -m models.lstmv5 --mode train --max_steps 1 --max_epochs 1 --max_patients 100 --learning_rate 1e-4 --batch_size 32 --timeseries_model multiLSTM --mini --dataset mimic_demo
# python -m models.lstmv5 --mode train --max_steps 1 --max_epochs 1 --max_patients 500 --learning_rate 1e-4 --batch_size 32 --timeseries_model singleLSTM --mini --dataset miiv 
# python -m models.lstmv5 --mode train --max_steps 10000 --max_epochs 50 --max_patients 10000 --learning_rate 1e-4 --batch_size 32 --timeseries_model singleLSTM --mini --dataset miiv 
# data/processed/step_5_miiv/miiv_processed_data_small.pkl


# debug 
# srun --time=1:00:00 --mem=120G --cpus-per-task=12 --gres=gpu:1 --partition=shared-gpu python -m models.lstmv5 --dataset eicu --max_steps 10 --max_patients 1000 --batch_size 32 --timeseries_model singleLSTM --mini

# srun --time=1:00:00 --mem=120G --cpus-per-task=12 --gres=gpu:1 --partition=shared-gpu python -m models.lstmv5 --dataset miiv --max_steps 10 --max_patients 20000 --batch_size 32 --timeseries_model singleLSTM --mini

# srun --time=10:00:00 --mem=120G --cpus-per-task=12 --gres=gpu:2 --partition=shared-gpu  python -m models.lstmv5 --dataset mimic --mode train --max_steps 500 --max_epochs 5 --max_patients 10000 --learning_rate 0.007  --batch_size 
# python -m models.lstmv5 --dataset sic --mode train --max_steps 500 --max_epochs 5 --max_patients 10000 --learning_rate 0.007  --batch_size 4 --timeseries_model singleLSTM