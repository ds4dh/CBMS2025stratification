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

class ChannelWiseLSTM(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_channels,
        optimizer_type='AdamW',
        learning_rate=5e-3,
        timeseries_mode='single',  # 'single' or 'multi'
        # dropout_rate=0.2
    ):
        super(ChannelWiseLSTM, self).__init__()
        self.save_hyperparameters()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        # Channel-wise LSTMs
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=False,
                # dropout=dropout_rate
            ) for _ in range(num_channels)
        ])
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * num_channels, output_dim)

        # Loss function
        self.criterion = nn.MSELoss(reduction='mean')  # Use 'none' to compute per-element loss

        # Initialize missing_features as a ParameterList aligned with channels
        self.missing_features = nn.ParameterList([
            nn.Parameter(torch.randn(1)) for _ in range(num_channels)
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
        # Replace missing values (-100) with learned missing feature values
        missing_features_tensor = torch.stack(
            [feat for feat in self.missing_features],
            dim=-1,
        ).view(1, 1, -1)  # (1, 1, num_channels)
        ts_data = torch.where(ts_data == -100, missing_features_tensor, ts_data)
        # if there is 0 dimension in timesteps put also missing_features_tensor (to-be-optimized imputation per chnnel parameter)
        if ts_data.size(1) == 0:
            ts_data = missing_features_tensor
            lengths = torch.tensor([1])

        # Initialize hidden and cell states
        h_n_list = [None] * self.num_channels
        c_n_list = [None] * self.num_channels
        
        batch_size = ts_data.size(0)
        lstm_channel_outputs = []
        
        for i in range(self.num_channels):
            # Extract i-th channel data
            channel_data = ts_data[:, :, i].unsqueeze(-1)  # (batch_size, seq_len, 1)

            if channel_data.size(1) == 0: 
                # Append a tensor of zeros with the correct shape (batch_size, 0, hidden_dim) 
                lstm_channel_outputs.append(torch.zeros(batch_size, 0, self.hidden_dim, device=ts_data.device))
                # Append zeros for h_n and c_n with the correct shape (batch_size, hidden_dim)
                h_n_list.append(torch.zeros(batch_size, self.hidden_dim, device=ts_data.device))
                c_n_list.append(torch.zeros(batch_size, self.hidden_dim, device=ts_data.device))
                continue 
            # Pack the sequence
            try:
                packed_channel_data = pack_padded_sequence(
                    channel_data.contiguous(), lengths.cpu(), batch_first=True, enforce_sorted=False
                )
            except:
                import pdb; pdb.set_trace()
            # Initialize hidden and cell states if not already
            if h_n_list[i] is None:
                # print('hidden state initialization')
                h_n_list[i] = torch.zeros(1, batch_size, self.hidden_dim, device=ts_data.device)
                c_n_list[i] = torch.zeros(1, batch_size, self.hidden_dim, device=ts_data.device)
            
            # LSTM computation
            packed_lstm_out, (h_n_list[i], c_n_list[i]) = self.lstms[i](
                packed_channel_data, (h_n_list[i], c_n_list[i])
            )
            
            # Unpack output sequence
            channel_lstm_output, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)
            lstm_channel_outputs.append(channel_lstm_output)
        
        # Concatenate the outputs from all channels along the feature dimension
        final_lstm_output = torch.cat(lstm_channel_outputs, dim=2)  # (batch, seq_len, hidden_dim * num_channels)
        
        # Pass through FC layer
        final_output = self.fc(final_lstm_output)  # (batch, seq_len, output_dim)
        
        # Aggregate hidden states per patient
        # Each h_n_list[i] has shape (1, batch_size, hidden_dim)
        # Squeeze the first dimension to remove it
        try:
            h_n_list = [h.squeeze(0) for h in h_n_list]  # Each h now has shape (batch_size, hidden_dim)
        except:
            import pdb; pdb.set_trace()

        # Stack along the channel dimension to get (batch_size, num_channels, hidden_dim)
        stacked_h_n = torch.stack(h_n_list, dim=1)  # (batch_size, num_channels, hidden_dim)
        
        # Flatten to (batch_size, num_channels * hidden_dim)
        last_hidden_states = stacked_h_n.reshape(batch_size, -1)  # (batch_size, num_channels * hidden_dim)
        
        return final_output, last_hidden_states
    
    def compute_loss(self, batch):
        
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
        
        # Get model outputs
        outputs, _ = self(inputs_padded, lengths)

        # Filter out padded/missing target values
        valid_mask_targets = (targets_padded != -100) & (targets_padded != -111)
        meaningful_outputs = outputs[valid_mask_targets]
        meaningful_targets = targets_padded[valid_mask_targets]

        # # Check for NaNs
        # if torch.isnan(meaningful_outputs).any() or torch.isnan(meaningful_targets).any():
        #     print("NaN detected in outputs or targets")
        #     import pdb; pdb.set_trace()

        # Calculate loss
        res = self.criterion(meaningful_outputs, meaningful_targets)
        return res

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
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
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        # /home/dproios/.local/share/virtualenvs/ricu_clean-XEyIo3Un/lib/python3.9/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
        lrtolog = scheduler.get_last_lr()
        print(f"Learning rate: {lrtolog}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Metric to monitor for scheduling
                'interval': 'epoch',    # Apply scheduler every epoch
                'frequency': 1
            }
        }