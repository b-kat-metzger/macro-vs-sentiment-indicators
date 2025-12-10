"""LSTM regressors for economic and financial feature sets.

Loads `src/data/standardized_monthly_data.csv`, builds sequence data,
trains two LSTM models (one using economic features, one using financial
features) and reports R^2, RMSE and MAE on a holdout test set.

Usage: run the file as a script to train and evaluate both models.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

tf.get_logger().setLevel('ERROR')


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def feature_sets_from_df(df: pd.DataFrame) -> dict:
    econ = [c for c in df.columns if c.startswith("ECO") or c.startswith("FRED_")]
    fin = [c for c in df.columns if c.startswith("FIN_") and c != "FIN_gspc_ret_lag1"]
    return {"econ": econ, "fin": fin}


def prepare_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    lookback: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn tabular features into LSTM sequences.

    For i in range(lookback, n): X[i] = features[i-lookback:i] and y[i] = target[i]
    Ensures target is 1D.
    """
    Xs = []
    ys = []
    arr = features.values.astype(float)
    targ = target.values.astype(float)
    
    # Ensure target is 1D
    if targ.ndim > 1:
        targ = targ.flatten()
    
    n = len(features)
    for i in range(lookback, n):
        Xs.append(arr[i - lookback : i])
        ys.append(targ[i])
    return np.array(Xs), np.array(ys)


def build_lstm_model(input_shape: tuple) -> tf.keras.Model:
    model = Sequential(
        [
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss="mse", metrics=["mae"])
    return model


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "FIN_gspc_ret_lag1",
    lookback: int = 12,
    test_fraction: float = 0.2,
    epochs: int = 150,
    batch_size: int = 8,
) -> dict:
    tf.random.set_seed(67)
    np.random.seed(67)

    # filter columns and drop rows without a target
    data = df[[*feature_cols, target_col]].copy()
    data = data.dropna(subset=[target_col]).ffill().bfill()

    X_seq, y = prepare_sequences(data[feature_cols], data[target_col], lookback=lookback)
    
    print(f"  Total sequences: {X_seq.shape[0]}")
    print(f"  Sequence shape: {X_seq.shape}")
    print(f"  Target range: [{y.min():.6f}, {y.max():.6f}], mean: {y.mean():.6f}, std: {y.std():.6f}")

    # chronological train/test split
    split = int(len(X_seq) * (1 - test_fraction))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"  y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    # Flatten train sequences to fit scaler then reshape
    nsamples, nt, nfeat = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape((nsamples * nt, nfeat))
    scaler.fit(X_train_flat)

    # apply scaler
    X_train_scaled = scaler.transform(X_train_flat).reshape((nsamples, nt, nfeat))
    X_test_flat = X_test.reshape((X_test.shape[0] * nt, nfeat))
    X_test_scaled = scaler.transform(X_test_flat).reshape((X_test.shape[0], nt, nfeat))
    
    # Also scale the target values
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    model = build_lstm_model(input_shape=(nt, nfeat))

    es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0)
    model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )

    # Predict on scaled data
    y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
    
    # Inverse transform predictions
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Final shape check
    y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test
    y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
    
    print(f"  y_pred shape: {y_pred_flat.shape}, y_test shape: {y_test_flat.shape}")
    print(f"  Pred range: [{y_pred_flat.min():.6f}, {y_pred_flat.max():.6f}], mean: {y_pred_flat.mean():.6f}")
    
    # Ensure shapes match
    if y_pred_flat.shape != y_test_flat.shape:
        raise ValueError(f"Prediction shape mismatch: y_pred={y_pred_flat.shape}, y_test={y_test_flat.shape}")
    
    r2 = r2_score(y_test_flat, y_pred_flat)
    rmse = math.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    
    # Baseline: persistence (predict same as previous)
    baseline_pred = np.concatenate([[y_train[-1]], y_test_flat[:-1]])
    baseline_r2 = r2_score(y_test_flat, baseline_pred)

    return {
        "model": model,
        "scaler": scaler,
        "y_scaler": y_scaler,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "baseline_r2": baseline_r2,
        "y_test": y_test_flat,
        "y_pred": y_pred_flat,
    }


def plot_predictions(results: dict, name: str, save_path: str = None):
    """Plot predictions vs actuals and save to file."""
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series plot
    ax = axes[0]
    ax.plot(y_test, label="Actual", linewidth=2, alpha=0.7)
    ax.plot(y_pred, label="Predicted", linewidth=2, alpha=0.7)
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Return")
    ax.set_title(f"{name}: Predictions vs Actuals (Time Series)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Scatter plot
    ax = axes[1]
    ax.scatter(y_test, y_pred, alpha=0.6, s=30)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual Return")
    ax.set_ylabel("Predicted Return")
    ax.set_title(f"{name}: Predictions vs Actuals (Scatter)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    plt.close()


def plot_aggregate_summary(econ_results: dict, fin_results: dict, save_path: str = None):
    """Create aggregate visualization comparing economic vs financial LSTM predictions.
    
    Parameters:
    -----------
    econ_results : dict
        Results from train_and_evaluate() for economic features
    fin_results : dict
        Results from train_and_evaluate() for financial features
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Economic features - Predicted vs Actual
    ax = axes[0]
    y_test_econ = econ_results['y_test']
    y_pred_econ = econ_results['y_pred']
    r2_econ = econ_results['r2']
    
    ax.plot(y_test_econ, 'o-', label='Actual', linewidth=2.5, markersize=5, color='#1f77b4', alpha=0.8)
    ax.plot(y_pred_econ, 's--', label='Predicted', linewidth=2.5, markersize=4, color='#ff7f0e', alpha=0.8)
    ax.fill_between(range(len(y_test_econ)), y_test_econ, y_pred_econ, alpha=0.1, color='gray')
    ax.set_title(f'Economic Features LSTM: Predicted vs Actual (R² = {r2_econ:.4f})', 
                fontweight='bold', fontsize=12)
    ax.set_xlabel('Test Sample Index', fontsize=11)
    ax.set_ylabel('S&P 500 Monthly Return', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Financial features - Predicted vs Actual
    ax = axes[1]
    y_test_fin = fin_results['y_test']
    y_pred_fin = fin_results['y_pred']
    r2_fin = fin_results['r2']
    
    ax.plot(y_test_fin, 'o-', label='Actual', linewidth=2.5, markersize=5, color='#1f77b4', alpha=0.8)
    ax.plot(y_pred_fin, 's--', label='Predicted', linewidth=2.5, markersize=4, color='#2ca02c', alpha=0.8)
    ax.fill_between(range(len(y_test_fin)), y_test_fin, y_pred_fin, alpha=0.1, color='gray')
    ax.set_title(f'Financial Features LSTM: Predicted vs Actual (R² = {r2_fin:.4f})', 
                fontweight='bold', fontsize=12)
    ax.set_xlabel('Test Sample Index', fontsize=11)
    ax.set_ylabel('S&P 500 Monthly Return', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    fig.suptitle('LSTM Models: Economic vs Financial Features Comparison', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Aggregate plot saved to {save_path}")
    plt.close()


def main():
    base = Path(__file__).resolve().parents[1] / "data" / "standardized_monthly_data.csv"
    csv_path = os.environ.get("STANDARDIZED_MONTHLY_DATA", str(base))
    df = load_data(csv_path)

    sets = feature_sets_from_df(df)
    target = "FIN_gspc_ret_lag1"

    print(f"Found feature sets: econ={len(sets['econ'])}, fin={len(sets['fin'])}")
    print(f"Data shape: {df.shape}")

    print("\n" + "="*60)
    print("Training economic-features LSTM...")
    print("="*60)
    econ_res = train_and_evaluate(df, sets["econ"], target_col=target)
    print("Economic features results:")
    print(f"  R²:          {econ_res['r2']:.4f}")
    print(f"  Baseline R²: {econ_res['baseline_r2']:.4f}")
    print(f"  RMSE:        {econ_res['rmse']:.6f}")
    print(f"  MAE:         {econ_res['mae']:.6f}")
    plot_predictions(econ_res, "Economic Features LSTM", "visuals/lstm_econ_predictions.png")

    print("\n" + "="*60)
    print("Training financial-features LSTM...")
    print("="*60)
    fin_res = train_and_evaluate(df, sets["fin"], target_col=target)
    print("Financial features results:")
    print(f"  R²:          {fin_res['r2']:.4f}")
    print(f"  Baseline R²: {fin_res['baseline_r2']:.4f}")
    print(f"  RMSE:        {fin_res['rmse']:.6f}")
    print(f"  MAE:         {fin_res['mae']:.6f}")
    plot_predictions(fin_res, "Financial Features LSTM", "visuals/lstm_fin_predictions.png")
    
    # print("\n" + "="*60)
    # print("Summary")
    # print("="*60)
    # print(f"Economic R²:          {econ_res['r2']:.4f}")
    # print(f"Economic Baseline R²: {econ_res['baseline_r2']:.4f}")
    # print(f"Financial R²:         {fin_res['r2']:.4f}")
    # print(f"Financial Baseline R²:{fin_res['baseline_r2']:.4f}")

    
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Metric':<30} {'Economic':<20} {'Financial':<20}")
    print("-" * 70)
    print(f"{'Test R²':<30} {econ_res['r2']:>18.4f} {fin_res['r2']:>18.4f}")
    print(f"{'Test RMSE':<30} {econ_res['rmse']:>18.6f} {fin_res['rmse']:>18.6f}")
    print(f"{'Test MAE':<30} {econ_res['mae']:>18.6f} {fin_res['mae']:>18.6f}")
    print("-" * 70)

    # Generate aggregate visualization
    print(f"\n{'='*70}")
    print("Generating aggregate visualization...")
    print(f"{'='*70}")
    plot_aggregate_summary(econ_res, fin_res, "visuals/lstm_aggregate_comparison.png")
    
    print("\nVisualization plots saved to visuals/")



if __name__ == "__main__":
    main()

