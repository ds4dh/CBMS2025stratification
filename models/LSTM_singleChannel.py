import time
from torch.optim.lr_scheduler import ReduceLROnPlateau  # or another scheduler of choice
import numpy as np
import pickle 
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl
from utils.config import ALL_FEATURES
import random 

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_float32_matmul_precision('medium')
ALL_FEATURES = [feat for feat in ALL_FEATURES if feat != "icds" and 'ws_' not in feat]

import gc
import torch
import pytorch_lightning as pl


class LSTM_singleChannel(pl.LightningModule):
    def __init__(
        self,
        input_dim=None,
        hidden_dim=None,
        output_dim=None,
        optimizer_type='AdamW',
        learning_rate=5e-3,
        timeseries_mode='single',
        lstmcell=True,
        num_LSTM_layers=2,
        num_post_layers=2,
    ):
        super(LSTM_singleChannel, self).__init__()
        self.save_hyperparameters()
        
        # Access hyperparameters from self.hparams
        input_dim = self.hparams.input_dim
        hidden_dim = self.hparams.hidden_dim
        output_dim = self.hparams.output_dim
        optimizer_type = self.hparams.optimizer_type
        learning_rate = self.hparams.learning_rate
        lstmcell = self.hparams.lstmcell
        num_LSTM_layers = self.hparams.num_LSTM_layers
        num_post_layers = self.hparams.num_post_layers
        timeseries_mode = self.hparams.timeseries_mode
        
        # Validate essential hyperparameters
        if input_dim is None or hidden_dim is None or output_dim is None:
            raise ValueError('input_dim, hidden_dim, and output_dim must be specified.')

        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.cell_type = lstmcell

        # Channel-wise LSTMs
        if lstmcell:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        else:
            self.lstm = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=False)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Loss function
        self.criterion = nn.MSELoss(reduction='mean')

        # Initialize missing_features as a ParameterList aligned with channels
        self.missing_features = nn.ParameterList([
            nn.Parameter(torch.randn(1)) for _ in range(input_dim)
        ])
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param) 
        
        
    def forward(self, ts_data, lengths):
        # Replace missing values
        missing_features_tensor = torch.stack(
            [feat for feat in self.missing_features],
            dim=-1,
        ).view(1, 1, -1)  # (1, 1, num_channels)
        ts_data = torch.where(ts_data == -100, missing_features_tensor, ts_data)
        # print("After missing_features_tensor torch.where, contiguous:", ts_data.is_contiguous())
        # Reshape to merge channels
        batch_size, seq_len, num_channels = ts_data.shape
        ts_data = ts_data.view(batch_size, seq_len, -1).contiguous()  # (batch_size, seq_len, input_dim * num_channels)
        
        # Pack sequence
        lengths = torch.full_like(lengths, lengths.max())
        ts_data= ts_data.contiguous()
        packed_data = pack_padded_sequence(ts_data, lengths.cpu(), batch_first=True, enforce_sorted=True)
        input_padded = ts_data.contiguous()
        # print(f"input_padded.is_contiguous(): {input_padded.is_contiguous()}")
        # print(f"lengths: {lengths}") # added print statement
        packed_data = torch.nn.utils.rnn.pack_padded_sequence(input_padded.contiguous(), lengths.cpu(), batch_first=True, enforce_sorted=False)
        # print(f"packed_data.data.is_contiguous(): {packed_data.data.is_contiguous()}")
        
        # packed_data = pack_padded_sequence(ts_data, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # print("Packed data batch_sizes:", packed_data.batch_sizes)
        # print("Packed data sorted_indices:", packed_data.sorted_indices)
        # print("Packed data unsorted_indices:", packed_data.unsorted_indices)
        # print("Packed data is_contiguous:", packed_data.data.is_contiguous())
        # Forward pass through single LSTM
        
        if self.cell_type: # lstm
            packed_output, (h_n, c_n) = self.lstm(packed_data)
        else: # gru
            packed_output, h_n = self.lstm(packed_data)
        # File "/home/users/p/proios0/work/ricu_clean/models/LSTM_singleChannel.py", line 77, in forward
        #     packed_output, (h_n, c_n) = self.lstm(packed_data.contiguous())
        # AttributeError: 'PackedSequence' object has no attribute 'contiguous'
        # srun: error: gpu013: task 0: Exited with exit code 1               
        
        # Unpack output
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Pass through FC layer
        final_output = self.fc(lstm_output)
        
        # Use the last hidden state for further operations
        last_hidden_state = h_n.squeeze(0).reshape(batch_size, -1)  # (batch_size, hidden_dim) # TODO check exactly why its needed 
        
        return final_output, last_hidden_state


    # # The default chunk length for TBPTT (you can also pass it in __init__).
    # def tbptt_split_batch(self, batch, split_size=100):
    #     """
    #     batch: (patient_ids, inputs_padded, targets_padded, lengths)
    #     Returns chunks of these tensors each of length 'split_size' along time dim.
    #     """
    #     patient_ids, inputs_padded, targets_padded, lengths = batch

    #     # inputs_padded shape: (B, T, input_dim)
    #     # targets_padded shape: (B, T, output_dim) or same shape as inputs.

    #     max_time = inputs_padded.size(1)  # T

    #     # We'll produce a list of smaller sub-batches, each up to 'split_size' timesteps
    #     for start in range(0, max_time, split_size):
    #         end = start + split_size

    #         # Sub-slice along time dimension
    #         x_chunk = inputs_padded[:, start:end, :]
    #         y_chunk = targets_padded[:, start:end, :]

    #         # Adjust chunked lengths. 
    #         # For each sample, the length in this chunk is min(original_length - start, split_size)
    #         chunk_lengths = torch.clamp(lengths - start, min=0, max=split_size)

    #         # Return a sub-batch with the same format
    #         yield (patient_ids, x_chunk, y_chunk, chunk_lengths)
    def compute_loss(self, batch):
        start_time = time.time()
        
        # Get input data
        patient_ids, inputs_padded, targets_padded, lengths = batch
        if inputs_padded is None: return None
        
        # Filter out zero-length samples
        if 0 in lengths:
            valid_indices = lengths > 0
            patient_ids = [pid for i, pid in enumerate(patient_ids) if valid_indices[i]]
            inputs_padded = inputs_padded[valid_indices]
            targets_padded = targets_padded[valid_indices]
            lengths = lengths[valid_indices]
        inputs_padded = inputs_padded.contiguous()

        # Get model outputs
        outputs, _ = self(inputs_padded.contiguous(), lengths)

        # Filter out padded/missing target values
        valid_mask_targets = (targets_padded != -100) & (targets_padded != -111)
        meaningful_outputs = outputs[valid_mask_targets]
        meaningful_targets = targets_padded[valid_mask_targets]

        # Check for NaNs
        if torch.isnan(meaningful_outputs).any() or torch.isnan(meaningful_targets).any():
            print("NaN detected in outputs or targets")
            import pdb; pdb.set_trace()

        # Calculate loss
        loss_start = time.time()
        res = self.criterion(meaningful_outputs, meaningful_targets)
        if res.isnan():
            # return neutral loss
            print("NaN detected in loss")
            return torch.tensor(0.0, requires_grad=True)
        return res

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        if loss is None: 
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # Log current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, prog_bar=True)


    def configure_optimizers(self):
        # Choose optimizer
        if self.optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        # Define scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Metric to monitor for scheduling
                'interval': 'epoch',    # Apply scheduler every epoch
                'frequency': 1
            }
        }