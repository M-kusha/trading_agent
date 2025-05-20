import os
from typing import Dict
import pandas as pd
import numpy as np

def load_data(data_dir: str = "data/processed") -> Dict[str, Dict[str, pd.DataFrame]]:
    instruments = {
        "XAU/USD": {"H1": "XAUUSD_H1_features.csv", "H4": "XAUUSD_H4_features.csv", "D1": "XAUUSD_D1_features.csv"},
        "EUR/USD": {"H1": "EURUSD_H1_features.csv", "H4": "EURUSD_H4_features.csv", "D1": "EURUSD_D1_features.csv"},
    }

    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol, files in instruments.items():
        data[symbol] = {}
        for tf, fname in files.items():
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"[load_data] required file '{path}' not found.")

            df = pd.read_csv(path, parse_dates=["time"])
            # basic sanity
            needed = {"close", "volatility"}
            missing = needed.difference(df.columns)
            if missing:
                raise ValueError(f"[load_data] '{path}' missing {missing}.")

            df.sort_values("time",  inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["close", "volatility"], inplace=True)

            # cast numerics to float32
            num_cols = df.select_dtypes(include="number").columns
            df[num_cols] = df[num_cols].astype(np.float32)
            df["volatility"] = df["volatility"].clip(lower=1e-7)

            # ——— NEW: drop **all** zero-variance numeric columns ———
            stds = df[num_cols].std()
            zero_cols = stds[stds == 0.0].index.tolist()
            if zero_cols:
                df.drop(columns=zero_cols, inplace=True)
                print(f"[load_data] dropped zero‐variance cols for {symbol}-{tf}: {zero_cols}")

            data[symbol][tf] = df

    return data
