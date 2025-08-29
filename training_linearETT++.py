# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import os
import random
from sklearn.preprocessing import StandardScaler
import copy
import sys

sys.path.append('/home/gsinger')
from samformer.samformer_pytorch.samformer.utils.sam import SAM
from samformer.samformer_pytorch.samformer.samformer import SAMFormer

def construct_sliding_window_data(data, seq_len, pred_len, time_increment=1):
    n_samples = data.shape[0] - (seq_len - 1) - pred_len
    range_ = np.arange(0, n_samples, time_increment)
    x, y = list(), list() 
    for i in range_:
        x.append(data[i:(i + seq_len)].T) 
        y.append(data[(i + seq_len):(i + seq_len + pred_len)].T) 
    return np.array(x), np.array(y)


def read_ETTh1_dataset(seq_len, pred_len, time_increment=1, file_name=None, etth=False, ettm=False):
    file_name = file_name
    df_raw = pd.read_csv(file_name, index_col=0)
    n = len(df_raw)
    if ettm:
        train_end = 12 * 30 * 24 *4
        val_end = train_end + 4 * 30 * 24*4
        test_end = val_end + 4 * 30 * 24*4
    if etth==True:
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = val_end + 4 * 30 * 24
    else:
        train_end = int(n * 0.7)
        val_end = n - int(n * 0.2)
        test_end = n
        
    train_df = df_raw[:train_end]
    val_df = df_raw[train_end - seq_len : val_end]
    test_df = df_raw[val_end - seq_len : test_end]

    scaler = StandardScaler()
    scaler.fit(train_df.values)
    train_df, val_df, test_df = [scaler.transform(df.values) for df in [train_df, val_df, test_df]]
    
    x_train, y_train = construct_sliding_window_data(train_df, seq_len, pred_len, time_increment)
    x_val, y_val = construct_sliding_window_data(val_df, seq_len, pred_len, time_increment)
    x_test, y_test = construct_sliding_window_data(test_df, seq_len, pred_len, time_increment)
    
    print(f"Dataset shapes: x_train={x_train.shape}, y_train={y_train.shape}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) https://openreview.net/pdf?id=cGDAkQo1C0p
    https://github.com/ts-kim/RevIN
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class LabeledDataset(Dataset):
    def __init__(self, x, y):

        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        
        print(f"Dataset x shape: {self.x.shape}, y shape: {self.y.shape}")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# %%
import pandas as pd 
hyperopt_params=pd.read_csv('/home/gsinger/SamformerDistill/linearWalid/hp_selected1.csv')
class LinearForecaster2(nn.Module):
    """
    Simple per-channel linear forecaster with optional RevIN.
    SAMFormer will instantiate this with (num_channels, seq_len, pred_horizon, use_revin).
    """
    def __init__(self, num_channels, seq_len, pred_horizon, use_revin=True):
        super().__init__()
        self.revin = RevIN(num_features=num_channels)
        self.linear_forecaster = nn.Linear(seq_len, pred_horizon)
        self.use_revin = use_revin

    def forward(self, x, flatten_output=False):
        # x: (N, D, L)
        if self.use_revin:
            x = self.revin(x.transpose(1, 2), mode="norm").transpose(1, 2)  # (N, D, L)
        out = self.linear_forecaster(x)  # (N, D, H)
        if self.use_revin:
            out = self.revin(out.transpose(1, 2), mode="denorm").transpose(1, 2)
        return out


# %%
import pandas as pd
import numpy as np
import torch

# Read the hyperparameters CSV
hparams_df = pd.read_csv('/home/gsinger/SamformerDistill/linearWalid/hp_selected1.csv') 

# Initialize results list
results = []

# Loop through each row in the hyperparameters dataframe
for idx, row in hparams_df.iterrows():
    dataset = row['dataset']
    file_path = row['file']
    horizon = row['horizon']
    seq_len = row['seq_len']
    learning_rate = row['learning_rate']
    weight_decay = row['weight_decay']
    rho = row['rho']
    batch_size = row['batch_size']
    use_revin = row['use_revin']
    num_epochs = row['num_epochs']
    seed = row['seed']
    device = row['device']
    
    print(f"Processing {dataset}, horizon={horizon}")
    
    try:
        # Determine dataset type flags
        etth_flag = dataset.startswith('ETTh')
        ettm_flag = dataset.startswith('ETTm')
        weather_flag = dataset == 'Weather'
        exchange_flag = dataset == 'Exchange'
        
        # Load dataset
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_ETTh1_dataset(
            file_name=file_path, 
            seq_len=seq_len, 
            pred_len=horizon,
            etth=etth_flag,
            ettm=ettm_flag
        )

        model = SAMFormer(
            network=LinearForecaster2,
            device=device,
            num_epochs=num_epochs,
            batch_size=batch_size,
            base_optimizer=torch.optim.Adam,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            rho=rho,
            use_revin=use_revin,
        )
        
        model.fit(x_train, y_train)
        
        # Make predictions
        y_pred = model.predict(x_test).astype(np.float32)
        
        # Calculate MAE
        mse = np.mean(np.abs(y_test - y_pred)**2)
        
        # Store results
        results.append({
            'dataset': dataset,
            'horizon': horizon,
            'seq_len': seq_len,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'rho': rho,
            'batch_size': batch_size,
            'use_revin': use_revin,
            'num_epochs': num_epochs,
            'seed': seed,
            'device': device,
            'mae': mse
        })
        
        print(f"Completed {dataset}, horizon={horizon}, MSE={mse:.3f}")
        
    except Exception as e:
        print(f"Error processing {dataset}, horizon={horizon}: {str(e)}")
        # Still record the error case
        results.append({
            'dataset': dataset,
            'horizon': horizon,
            'seq_len': seq_len,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'rho': rho,
            'batch_size': batch_size,
            'use_revin': use_revin,
            'num_epochs': num_epochs,
            'seed': seed,
            'device': device,
            'mae': np.nan,
            'error': str(e)
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('/home/gsinger/SamformerDistill/linearWalid/experiments/mse_results.csv', index=False)

print("Experiment completed. Results saved to mae_results.csv")

# Display results
print("\nMAE Results Summary:")
print(results_df[['dataset', 'horizon', 'mae']]) 

# %%
import pandas as pd
import numpy as np
import torch

# Read the hyperparameters CSV
hparams_df = pd.read_csv('/home/gsinger/SamformerDistill/linearWalid/hp_selected1.csv') 

# Initialize results list
results = []

# Loop through each row in the hyperparameters dataframe
for idx, row in hparams_df.iterrows():
    dataset = row['dataset']
    file_path = row['file']
    horizon = row['horizon']
    seq_len = row['seq_len']
    learning_rate = row['learning_rate']
    weight_decay = row['weight_decay']
    rho = row['rho']
    batch_size = row['batch_size']
    use_revin = row['use_revin']
    num_epochs = row['num_epochs']
    seed = row['seed']
    device = row['device']
    
    print(f"Processing {dataset}, horizon={horizon}")
    
    try:
        # Determine dataset type flags
        etth_flag = dataset.startswith('ETTh')
        ettm_flag = dataset.startswith('ETTm')
        weather_flag = dataset == 'Weather'
        exchange_flag = dataset == 'Exchange'
        
        # Load dataset
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_ETTh1_dataset(
            file_name=file_path, 
            seq_len=seq_len, 
            pred_len=horizon,
            etth=etth_flag,
            ettm=ettm_flag
        )

        model = SAMFormer(
            network=LinearForecaster2,
            device=device,
            num_epochs=num_epochs,
            batch_size=batch_size,
            base_optimizer=torch.optim.Adam,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            rho=rho,
            use_revin=use_revin,
        )
        
        model.fit(x_train, y_train)
        
        # Make predictions
        y_pred = model.predict(x_test).astype(np.float32)
        
        # Calculate MAE
        mse = np.mean(np.abs(y_test - y_pred)**2)
        
        # Store results
        results.append({
            'dataset': dataset,
            'horizon': horizon,
            'seq_len': seq_len,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'rho': rho,
            'batch_size': batch_size,
            'use_revin': use_revin,
            'num_epochs': num_epochs,
            'seed': seed,
            'device': device,
            'mae': mse
        })
        
        print(f"Completed {dataset}, horizon={horizon}, MSE={mse:.3f}")
        
    except Exception as e:
        print(f"Error processing {dataset}, horizon={horizon}: {str(e)}")
        # Still record the error case
        results.append({
            'dataset': dataset,
            'horizon': horizon,
            'seq_len': seq_len,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'rho': rho,
            'batch_size': batch_size,
            'use_revin': use_revin,
            'num_epochs': num_epochs,
            'seed': seed,
            'device': device,
            'mae': np.nan,
            'error': str(e)
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('/home/gsinger/SamformerDistill/linearWalid/experiments/mse_results.csv', index=False)

print("Experiment completed. Results saved to mae_results.csv")

# Display results
print("\nMAE Results Summary:")
print(results_df[['dataset', 'horizon', 'mae']]) 


