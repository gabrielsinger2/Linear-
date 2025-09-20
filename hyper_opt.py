#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Singer Gabriel
"""
import os
import sys
import gc
import time
import copy
import json
import math
import random
import argparse
from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

DEFAULT_REPO_ROOT = "/home/gsinger"
if DEFAULT_REPO_ROOT not in sys.path:
    sys.path.append(DEFAULT_REPO_ROOT)

from samformer.samformer_pytorch.samformer.samformer import SAMFormer  # noqa: E402


def set_seeds(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # better speed
    torch.backends.cudnn.benchmark = True


def construct_sliding_window_data(data, seq_len, pred_len, time_increment=1):
    """
    data: array (T, D)
    returns:
       x: (N, D, L), y: (N, D, H)
    """
    n_samples = data.shape[0] - (seq_len - 1) - pred_len
    idxs = np.arange(0, n_samples, time_increment)
    x, y = [], []
    for i in idxs:
        x.append(data[i:(i + seq_len)].T)                 # (D, L)
        y.append(data[(i + seq_len):(i + seq_len + pred_len)].T)  # (D, H)
    return np.array(x), np.array(y)


def read_ETT_dataset(file_name, seq_len, pred_len, time_increment=1, etth=False, ettm=False):
    """
    Splits ETTh/ETTm style (12m train / 4m val / 4m test) if flags are set; otherwise 70/10/20%.
    Standardizes by fitting scaler on train only.
    Returns windows for (train/val/test) + the scaler (in case you need it).
    """
    df_raw = pd.read_csv(file_name, index_col=0)
    n = len(df_raw)

    if ettm:
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = val_end + 4 * 30 * 24 * 4
    elif etth:
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = val_end + 4 * 30 * 24
    else:
        train_end = int(n * 0.7)
        val_end = n - int(n * 0.2)
        test_end = n

    train_df = df_raw.iloc[:train_end].copy()
    val_df   = df_raw.iloc[train_end - seq_len:val_end].copy()
    test_df  = df_raw.iloc[val_end - seq_len:test_end].copy()

    scaler = StandardScaler()
    scaler.fit(train_df.values)

    train_z = scaler.transform(train_df.values)
    val_z   = scaler.transform(val_df.values)
    test_z  = scaler.transform(test_df.values)

    x_train, y_train = construct_sliding_window_data(train_z, seq_len, pred_len, time_increment)
    x_val,   y_val   = construct_sliding_window_data(val_z,   seq_len, pred_len, time_increment)
    x_test,  y_test  = construct_sliding_window_data(test_z,  seq_len, pred_len, time_increment)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), scaler


# -------------------------
# Models (RevIN + Linear head)
# -------------------------

import torch.nn as nn

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

class LinearForecaster2(nn.Module):
    """
    Simple per-channel linear forecaster with optional RevIN.
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


# -------------------------
# Training & evaluation
# -------------------------

def mse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def fit_and_eval_once(
    x_train, y_train, x_val, y_val,
    device, num_epochs, batch_size, base_optim, learning_rate, weight_decay, rho, use_revin
):
    """
    Train a single model config on train, then eval on val (returns val MSE and model handle).
    """
    model = SAMFormer(
        network=LinearForecaster2,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        base_optimizer=base_optim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        rho=rho,
        use_revin=use_revin,
    )

    model.fit(x_train, y_train)
    with torch.no_grad():
        y_val_pred = model.predict(x_val)
    val_mse = mse_np(y_val, y_val_pred)
    return val_mse, model


def retrain_and_test(
    x_trainval, y_trainval, x_test, y_test,
    device, num_epochs, batch_size, base_optim, learning_rate, weight_decay, rho, use_revin
):
    """
    Retrain best config on train+val, evaluate on test.
    """
    model = SAMFormer(
        network=LinearForecaster2,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        base_optimizer=base_optim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        rho=rho,
        use_revin=use_revin,
    )

    model.fit(x_trainval, y_trainval)
    with torch.no_grad():
        y_test_pred = model.predict(x_test)
    test_mse = mse_np(y_test, y_test_pred)
    return test_mse


# -------------------------
# Main search routine
# -------------------------

def run_search_for_one(
    dataset_name: str,
    file_path: str,
    horizon: int,
    etth: bool,
    ettm: bool,
    device: str,
    out_dir: str,
    grid: dict,
    num_epochs: int,
    base_optim_ctor,
    seed: int,
):
    """
    Runs the grid search for a single dataset × horizon, appends CSVs as it goes.
    """
    os.makedirs(out_dir, exist_ok=True)
    search_log_csv = os.path.join(out_dir, "hp_search_log.csv")
    selected_csv   = os.path.join(out_dir, "hp_selected.csv")

    tried_rows = []
    t0_ds = time.time()
    best = {"val_mse": math.inf, "cfg": None}

    # We iterate seq_len in the grid; for each candidate, rebuild windows with that seq_len.
    for seq_len in grid["seq_len"]:
        # Build (train/val/test) slices for this seq_len + horizon
        (x_tr, y_tr), (x_va, y_va), (x_te, y_te), _ = read_ETT_dataset(
            file_name=file_path, seq_len=seq_len, pred_len=horizon,
            time_increment=grid.get("time_increment", 1),
            etth=etth, ettm=ettm
        )

        for (lr, wd, rho, batch_size, use_revin) in product(
            grid["learning_rate"], grid["weight_decay"], grid["rho"], grid["batch_size"], grid["use_revin"]
        ):
            cfg = dict(
                dataset=dataset_name,
                file=file_path,
                horizon=horizon,
                seq_len=seq_len,
                learning_rate=lr,
                weight_decay=wd,
                rho=rho,
                batch_size=batch_size,
                use_revin=bool(use_revin),
                num_epochs=num_epochs,
                seed=seed,
                device=device,
            )

            set_seeds(seed)
            torch.cuda.empty_cache()

            t0 = time.time()
            try:
                val_mse, _ = fit_and_eval_once(
                    x_tr, y_tr, x_va, y_va,
                    device=device,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    base_optim=base_optim_ctor,
                    learning_rate=lr,
                    weight_decay=wd,
                    rho=rho,
                    use_revin=bool(use_revin),
                )
            except Exception as e:
                # Log failures so you can spot bad configs
                cfg_row = copy.deepcopy(cfg)
                cfg_row.update(dict(val_mse=np.nan, seconds=time.time() - t0, error=str(e)))
                tried_rows.append(cfg_row)
                # flush partial log to disk
                pd.DataFrame([cfg_row]).to_csv(search_log_csv, mode="a", header=not os.path.exists(search_log_csv), index=False)
                print(f"[WARN] Failed config {cfg_row} — {e}")
                continue

            secs = time.time() - t0
            row = copy.deepcopy(cfg)
            row.update(dict(val_mse=val_mse, seconds=secs))
            tried_rows.append(row)
            # Append to search log CSV immediately (robust to interruptions)
            pd.DataFrame([row]).to_csv(search_log_csv, mode="a", header=not os.path.exists(search_log_csv), index=False)

            if val_mse < best["val_mse"]:
                best = {"val_mse": val_mse, "cfg": copy.deepcopy(cfg)}
                print(f"[{dataset_name} | H={horizon}] New best (val MSE={val_mse:.6f}) -> {cfg}")

            # Clean up
            del row
            gc.collect()
            torch.cuda.empty_cache()

        # Clean up per-seq_len data
        del x_tr, y_tr, x_va, y_va, x_te, y_te
        gc.collect()
        torch.cuda.empty_cache()

    # --------------- Final retrain on train+val with best hyperparams ---------------
    assert best["cfg"] is not None, f"No successful trials for {dataset_name} Horizon={horizon}"

    # Rebuild windows for the chosen seq_len to get train/val/test
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te), _ = read_ETT_dataset(
        file_name=file_path, seq_len=best["cfg"]["seq_len"], pred_len=horizon,
        time_increment=grid.get("time_increment", 1), etth=etth, ettm=ettm
    )
    # Merge train and val windows for final training (simple concat; no test leakage)
    x_trval = np.concatenate([x_tr, x_va], axis=0)
    y_trval = np.concatenate([y_tr, y_va], axis=0)

    # Retrain best config on (train+val), evaluate on test
    set_seeds(seed)
    test_mse = retrain_and_test(
        x_trval, y_trval, x_te, y_te,
        device=best["cfg"]["device"],
        num_epochs=best["cfg"]["num_epochs"],
        batch_size=best["cfg"]["batch_size"],
        base_optim=base_optim_ctor,
        learning_rate=best["cfg"]["learning_rate"],
        weight_decay=best["cfg"]["weight_decay"],
        rho=best["cfg"]["rho"],
        use_revin=best["cfg"]["use_revin"],
    )

    selected_row = copy.deepcopy(best["cfg"])
    selected_row.update(
        dict(
            selected_by_val_mse=best["val_mse"],
            test_mse=test_mse,
            total_seconds=time.time() - t0_ds,
        )
    )
    pd.DataFrame([selected_row]).to_csv(selected_csv, mode="a", header=not os.path.exists(selected_csv), index=False)

    print(f"[DONE] {dataset_name} | H={horizon} | best val MSE={best['val_mse']:.6f} | test MSE={test_mse:.6f}")
    return tried_rows, selected_row


def main():
    parser = argparse.ArgumentParser(description="Validation-based hyperopt for SAMFormer + LinearForecaster2")
    parser.add_argument("--datasets", nargs="+", default=["ETTm1", "ETTm2"], help="Datasets to run (keys of dataset_configs)")
    parser.add_argument("--out_dir", type=str, default=".", help="Directory to write CSV results")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size grid (single value)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--repo_root", type=str, default=DEFAULT_REPO_ROOT, help="Path added to sys.path for local imports")
    args = parser.parse_args()

    # Ensure repo root is in sys.path
    if args.repo_root not in sys.path:
        sys.path.append(args.repo_root)

    # Map dataset short names to csv paths + split style
    dataset_configs = {
        "ETTm1": dict(file="/home/gsinger/samformer/dataset/ETTm1.csv", etth=False, ettm=True),
        "ETTm2": dict(file="/home/gsinger/samformer/dataset/ETTm2.csv", etth=False, ettm=True),
        }

    # Hyperparameter grid (edit freely)
    grid = {
        "seq_len":       [128, 256, 512, 768],
        "learning_rate": [2e-4, 5e-4, 1e-3],
        "weight_decay":  [1e-6],
        "rho":           [0.0],             # SAM rho; 0.0 equals plain Adam if SAMFormer interprets it that way
        "batch_size":    [256],
        "use_revin":     [False],
        "time_increment": 1,
    }
    if args.batch_size is not None:
        grid["batch_size"] = [args.batch_size]

    base_optim_ctor = torch.optim.Adam

    # Create out_dir and echo config
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(
            dict(
                args=vars(args),
                grid=grid,
                dataset_configs={k: v for k, v in dataset_configs.items() if k in args.datasets},
            ),
            f,
            indent=2,
        )

    # Run
    for ds in args.datasets:
        assert ds in dataset_configs, f"Unknown dataset key: {ds}"
        file_path = dataset_configs[ds]["file"]
        etth = dataset_configs[ds]["etth"]
        ettm = dataset_configs[ds]["ettm"]
        assert os.path.exists(file_path), f"Missing dataset file at {file_path}"

        for horizon in args.horizons:
            try:
                run_search_for_one(
                    dataset_name=ds,
                    file_path=file_path,
                    horizon=horizon,
                    etth=etth,
                    ettm=ettm,
                    device=args.device,
                    out_dir=args.out_dir,
                    grid=grid,
                    num_epochs=args.epochs,
                    base_optim_ctor=base_optim_ctor,
                    seed=args.seed,
                )
            except AssertionError as e:
                print(f"[SKIP] {ds} | H={horizon} — {e}")
            except Exception as e:
                print(f"[ERROR] {ds} | H={horizon} — {e}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
