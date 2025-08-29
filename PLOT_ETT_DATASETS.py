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
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), scaler #add scaler for plot !

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


# %%


# %%
# raw.head()

# %%
# CSV = "/home/gsinger/samformer/dataset/exchange_rate.csv"
# raw = pd.read_csv(CSV)
# VALUE_COLS = raw.columns.values.tolist()[1:]

# %%


# %%

# import warnings, pandas as pd, numpy as np, torch, matplotlib.pyplot as plt
# from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# warnings.filterwarnings("ignore")

try:
    from autogluon.timeseries.models.chronos.model import ChronosModel
    def _load_model_pipeline_no_opt(self, is_training: bool = False):
        from autogluon.timeseries.models.chronos.pipeline import BaseChronosPipeline
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model_pipeline = BaseChronosPipeline.from_pretrained(
            self.model_path,
            device_map=device,
            torch_dtype=self.torch_dtype,
        )
    ChronosModel.load_model_pipeline = _load_model_pipeline_no_opt
except Exception:
    pass 

# CSV = "/home/gsinger/samformer/dataset/exchange_rate.csv"
# PRED_LEN = 96
# SEQ_LEN  = 512
# raw = pd.read_csv(CSV)
# VALUE_COLS = raw.columns.values.tolist()[1:]
# ITEMS_TO_PLOT = VALUE_COLS  

# # raw["timestamp"] = pd.to_datetime(raw["date"])
# # df_long = (
# #     raw.melt(id_vars="timestamp", value_vars=VALUE_COLS,
# #              var_name="item_id", value_name="target")
# #       .sort_values(["item_id","timestamp"])
# #       .loc[:, ["item_id","timestamp","target"]]
# # )

# # ts = TimeSeriesDataFrame.from_data_frame(df_long)

# # train_ts, test_ts = ts.train_test_split(PRED_LEN)



# %%
# import datasets

# ds = datasets.load_dataset("autogluon/chronos_datasets_extra", "ETTm", keep_in_memory=False, split="train")
# def to_pandas(ds: datasets.Dataset) -> "pd.DataFrame":
#     """Convert dataset to long data frame format."""
#     sequence_columns = [col for col in ds.features if isinstance(ds.features[col], datasets.Sequence)]
#     return ds.to_pandas().explode(sequence_columns).infer_objects()
# ds= to_pandas(ds)

# %%
import datasets
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

try:
    from autogluon.timeseries.models.chronos.model import ChronosModel
    def _load_model_pipeline_no_opt(self, is_training: bool = False):
        from autogluon.timeseries.models.chronos.pipeline import BaseChronosPipeline
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model_pipeline = BaseChronosPipeline.from_pretrained(
            self.model_path,
            device_map=device,
            torch_dtype=self.torch_dtype,
        )
    ChronosModel.load_model_pipeline = _load_model_pipeline_no_opt
except Exception:
    pass 



# ds = datasets.load_dataset("autogluon/chronos_datasets_extra", "ETTh", keep_in_memory=False, split="train")

# def to_pandas(ds: datasets.Dataset) -> "pd.DataFrame":
#     """Convert dataset to long data frame format."""
#     sequence_columns = [col for col in ds.features if isinstance(ds.features[col], datasets.Sequence)]
#     return ds.to_pandas().explode(sequence_columns).infer_objects()

# df = to_pandas(ds)

# # Filter for only the OT variable and prepare for AutoGluon
# df_ot = df[['id', 'timestamp', 'OT']].copy()
# df_ot = df_ot.rename(columns={'OT': 'target', 'id': 'item_id'})

# # Convert to TimeSeriesDataFrame
# ts_data = TimeSeriesDataFrame.from_data_frame(df_ot)

# # Split data
# PRED_LEN = 96
# train_data, test_data = ts_data.train_test_split(PRED_LEN)

# predictor = TimeSeriesPredictor(prediction_length=PRED_LEN, target='target').fit(
#     train_data,
#     presets="bolt_base",  
# )

# # Make predictions
# pred_ts = predictor.predict(test_data)
# import math
# import matplotlib.pyplot as plt
# import numpy as np

# pred_mean_etth1 = pred_ts['mean'].xs('ETTh1', level='item_id')
# chronos_all = pred_mean_etth1.to_numpy()



# def inverse_scale_windows(arr: np.ndarray, scaler) -> np.ndarray:
#     means  = scaler.mean_[None, :, None]   # (1, D, 1)
#     scales = scaler.scale_[None, :, None]  # (1, D, 1)
#     return arr * scales + means

# # Inverse scale predictions
# y_pred_orig = inverse_scale_windows(y_pred_sam, scaler)
# y_test_orig = inverse_scale_windows(y_test, scaler)

# idx=0
# plt.plot(y_pred_orig[idx,-1,:], label='Linear++')
# plt.plot(y_test_orig[idx,-1,:], label='Ground Truth')
# plt.plot(chronos_all[:PRED_LEN], label='Chronos Prediction')
# plt.legend()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import torch

# Configuration
PRED_LEN = 96
SEQ_LEN = 512

# Function to inverse scale predictions
def inverse_scale_windows(arr: np.ndarray, scaler) -> np.ndarray:
    means = scaler.mean_[None, :, None]   # (1, D, 1)
    scales = scaler.scale_[None, :, None]  # (1, D, 1)
    return arr * scales + means

# Function to process exchange_rate dataset
def process_exchange_rate():
    CSV = "/home/gsinger/samformer/dataset/exchange_rate.csv"
    
    # Load and preprocess data
    raw = pd.read_csv(CSV)
    VALUE_COLS = raw.columns.values.tolist()[1:]
    
    raw["timestamp"] = pd.to_datetime(raw["date"])
    df_long = (
        raw.melt(id_vars="timestamp", value_vars=VALUE_COLS,
                 var_name="item_id", value_name="target")
          .sort_values(["item_id","timestamp"])
          .loc[:, ["item_id","timestamp","target"]]
    )
    
    ts_data = TimeSeriesDataFrame.from_data_frame(df_long)
    train_ts, test_ts = ts_data.train_test_split(PRED_LEN)
    
    # Train Chronos model
    predictor = TimeSeriesPredictor(prediction_length=PRED_LEN, target='target').fit(
        train_ts,
        presets="bolt_base",  
    )
    
    # Make predictions with Chronos
    pred_ts = predictor.predict(test_ts)
    chronos_pred = pred_ts['mean'].values
    
    # Prepare data for SAMFormer
    (x_train, y_train), (x_val, y_val), (x_test, y_test), scaler = read_ETTh1_dataset(
        file_name=CSV, seq_len=SEQ_LEN, pred_len=PRED_LEN, etth=False, ettm=False
    )
    
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    
    # Train SAMFormer model
    model = SAMFormer(
        network=LinearForecaster2,
        device=("cuda:1" if torch.cuda.is_available() else "cpu"),
        num_epochs=100,
        batch_size=256,
        base_optimizer=torch.optim.Adam,
        learning_rate=1e-3,
        weight_decay=1e-6,
        rho=0,
        use_revin=False,
    )
    model.fit(x_train, y_train)
    
    # Make predictions with SAMFormer
    y_pred_sam = model.predict(x_test).astype(np.float32)
    y_pred_orig = inverse_scale_windows(y_pred_sam, scaler)
    y_test_orig = inverse_scale_windows(y_test, scaler)
    
    return y_test_orig, y_pred_orig, chronos_pred, scaler

# Function to process ETTh1 dataset
def process_etth1():
    from datasets import load_dataset
    import datasets
    
    # Load ETTh1 dataset
    ds = load_dataset("autogluon/chronos_datasets_extra", "ETTh", keep_in_memory=False, split="train")
    
    def to_pandas(ds: datasets.Dataset) -> "pd.DataFrame":
        """Convert dataset to long data frame format."""
        sequence_columns = [col for col in ds.features if isinstance(ds.features[col], datasets.Sequence)]
        return ds.to_pandas().explode(sequence_columns).infer_objects()
    
    df = to_pandas(ds)
    
    # Filter for only the OT variable and prepare for AutoGluon
    df_ot = df[['id', 'timestamp', 'OT']].copy()
    df_ot = df_ot.rename(columns={'OT': 'target', 'id': 'item_id'})
    
    # Convert to TimeSeriesDataFrame
    ts_data = TimeSeriesDataFrame.from_data_frame(df_ot)
    
    # Split data
    train_data, test_data = ts_data.train_test_split(PRED_LEN)
    
    # Train Chronos model
    predictor = TimeSeriesPredictor(prediction_length=PRED_LEN, target='target').fit(
        train_data,
        presets="bolt_base",  
    )
    
    # Make predictions with Chronos
    pred_ts = predictor.predict(test_data)
    pred_mean_etth1 = pred_ts['mean'].xs('ETTh1', level='item_id')
    chronos_pred = pred_mean_etth1.to_numpy()
    
    # Prepare data for SAMFormer
    CSV = "/home/gsinger/samformer/dataset/ETTh1.csv"  # Update with actual path
    (x_train, y_train), (x_val, y_val), (x_test, y_test), scaler = read_ETTh1_dataset(
        file_name=CSV, seq_len=SEQ_LEN, pred_len=PRED_LEN, etth=True, ettm=False
    )
    
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    
    # Train SAMFormer model
    model = SAMFormer(
        network=LinearForecaster2,
        device=("cuda:1" if torch.cuda.is_available() else "cpu"),
        num_epochs=100,
        batch_size=256,
        base_optimizer=torch.optim.Adam,
        learning_rate=1e-3,
        weight_decay=1e-6,
        rho=0,
        use_revin=False,
    )
    model.fit(x_train, y_train)
    
    # Make predictions with SAMFormer
    y_pred_sam = model.predict(x_test).astype(np.float32)
    y_pred_orig = inverse_scale_windows(y_pred_sam, scaler)
    y_test_orig = inverse_scale_windows(y_test, scaler)
    
    return y_test_orig, y_pred_orig, chronos_pred, scaler

# Main execution
if __name__ == "__main__":
    # Process both datasets
    print("Processing exchange_rate dataset...")
    y_test_exchange, y_pred_exchange, chronos_exchange, scaler_exchange = process_exchange_rate()
    
    print("Processing ETTh1 dataset...")
    y_test_etth1, y_pred_etth1, chronos_etth1, scaler_etth1 = process_etth1()
    
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot exchange_rate results
    ot_idx = -1  # OT variable index
    axes[0].plot(y_pred_exchange[0, ot_idx, :], label='SAMFormer (Linear++)')
    axes[0].plot(y_test_exchange[0, ot_idx, :], label='Ground Truth')
    axes[0].plot(chronos_exchange[:PRED_LEN], label='Chronos Prediction')
    axes[0].set_title('Exchange Rate Dataset - OT Variable')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot ETTh1 results
    axes[1].plot(y_pred_etth1[0, -1, :], label='SAMFormer (Linear++)')
    axes[1].plot(y_test_etth1[0, -1, :], label='Ground Truth')
    axes[1].plot(chronos_etth1[:PRED_LEN], label='Chronos Prediction')
    axes[1].set_title('ETTh1 Dataset - OT Variable')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison_plot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# %%
if __name__ == "__main__":
    # Process both datasets
    print("Processing exchange_rate dataset...")
    y_test_exchange, y_pred_exchange, chronos_exchange, scaler_exchange = process_exchange_rate()
    
    print("Processing ETTh1 dataset...")
    y_test_etth1, y_pred_etth1, chronos_etth1, scaler_etth1 = process_etth1()
    
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot exchange_rate results
    ot_idx = -1  # OT variable index
    axes[1].plot(y_pred_exchange[0, ot_idx, :], label='Linear++')
    axes[1].plot(y_test_exchange[0, ot_idx, :], label='Ground Truth')
    axes[1].plot(chronos_exchange[:PRED_LEN], label='Chronos_bolt')
    axes[1].set_title('Exchange Rate Dataset - OT Variable')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot ETTh1 results
    axes[0].plot(y_pred_etth1[0, -1, :], label='Linear++')
    axes[0].plot(y_test_etth1[0, -1, :], label='Ground Truth')
    axes[0].plot(chronos_etth1[:PRED_LEN], label='Chronos_bolt')
    axes[0].set_title('ETTh1 Dataset - OT Variable')
    axes[0].legend()
    axes[0].grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison_plot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Create subplots: 1 row, 2 cols
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # mettez sharey=True si vous voulez aligner les axes Y

# --- ETTh1 (gauche)
axes[0].plot(y_pred_etth1[0, -1, :], label='Linear++')
axes[0].plot(y_test_etth1[0, -1, :], label='Ground Truth')
axes[0].plot(chronos_etth1[:PRED_LEN], label='Chronos_bolt')
axes[0].set_title('ETTh1 Dataset')
axes[0].grid(True)

# --- Exchange Rate (droite)
ot_idx = -1  # OT variable index
idx=200
axes[1].plot(y_pred_exchange[idx, ot_idx, :], label='Linear++')
axes[1].plot(y_test_exchange[idx, ot_idx, :], label='Ground Truth')
axes[1].plot(chronos_exchange[:PRED_LEN], label='Chronos_bolt')
axes[1].set_title('Exchange Rate Dataset')
axes[1].grid(True)

handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
by_label = dict(zip(labels, handles))  # dédoublonne
fig.legend(by_label.values(), by_label.keys(),
           loc='lower center', ncol=3, frameon=True)

# Laisser de la place pour la légende en bas
fig.tight_layout(rect=(0, 0.12, 1, 1))  # réserve ~12% en bas
plt.savefig('comparison_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
import matplotlib.pyplot as plt

# Définir les paramètres de police globaux
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 17,
    'axes.labelsize': 17,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ot_idx = -1  # OT variable index
idx=500 #500 #300
line1, = axes[0].plot(y_pred_exchange[idx, ot_idx, :], label='SAMFormer (Linear++)', linewidth=2)
line2, = axes[0].plot(y_test_exchange[idx, ot_idx, :], label='Ground Truth', linewidth=2)
line3, = axes[0].plot(chronos_exchange[:PRED_LEN], label='Chronos Prediction', linewidth=2)
axes[0].set_title('Exchange Rate Dataset', fontsize=16, pad=15)
#axes[0].set_xlabel('Time Steps', fontsize=11)
#axes[0].set_ylabel('Value', fontsize=11)
axes[0].set_xticks([0,20,40,60,80,95])
axes[0].set_xticklabels([1,20,40,60,80,96])
axes[0].tick_params(axis='both', which='major', labelsize=12)
axes[0].grid(True, alpha=0.3)

# Plot ETTh1 results
axes[1].plot(y_pred_etth1[0, -1, :], linewidth=2)
axes[1].plot(y_test_etth1[0, -1, :], linewidth=2)
axes[1].plot(chronos_etth1[:PRED_LEN], linewidth=2)
axes[1].set_title('ETTh1 Dataset', fontsize=16, pad=15)
#axes[1].set_xlabel('Time Steps', fontsize=11)
#axes[1].set_ylabel('Value', fontsize=13)
axes[1].set_xticks([0,20,40,60,80,95])
axes[1].set_xticklabels([1,20,40,60,80,96])
axes[1].tick_params(axis='both', which='major', labelsize=12)
axes[1].grid(True, alpha=0.3)

# Ajouter une légende commune en bas de la figure
fig.legend([line1, line2, line3], 
           ['Linear++', 'Ground Truth', 'Chronos Prediction'],
           loc='lower center', 
           bbox_to_anchor=(0.5, 0), 
           ncol=3, 
           fontsize=15,
           frameon=True,
           fancybox=True,
           shadow=True,
           borderpad=1)

# Ajuster l'espacement pour faire de la place pour la légende en bas
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Ajuster cette valeur selon besoin

# Sauvegarder et afficher
plt.savefig('/home/gsinger/SamformerDistill/linearWalid/EX_ETTH_96.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%



