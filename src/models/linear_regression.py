"""
Linear regression models comparing economic vs financial indicators.
Predicts S&P 500 monthly returns using two separate feature sets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


# Feature groupings
ECONOMIC_FEATURES = [
    "ECO_dCPIAUCSL",
    "ECO_dFEDFUNDS",
    "ECO_dINDPRO",
    "FRED_UNRATE",
    "FRED_ICSA",
    "FRED_UMCSENT",
    "FRED_RSAFS",
    "FRED_AMTMNO",
    "FRED_PERMIT",
]

FINANCIAL_FEATURES = [
    "FIN_vix_level",
    "FIN_dVIX",
    "FIN_gspc_ret_lag1",
    "FIN_mom_12m",
    "FIN_mom_6m",
    "FIN_gspc_monthly_vol",
    "FIN_rv30",
    "FIN_yield_curve_slope",
    "FIN_credit_spread",
]


def load_and_prepare_data(csv_path: Path, train_test_split: float = 0.8):
    """
    Load data and prepare train/test sets.
    Target: S&P 500 monthly returns (derived from momentum features)
    
    Parameters:
    -----------
    csv_path : Path
        Path to standardized_monthly_data.csv
    train_test_split : float
        Fraction of data to use for training
    
    Returns:
    --------
    dict with keys: 'df', 'X_macro_train', 'X_macro_test', 'X_sentiment_train', 
                    'X_sentiment_test', 'y_train', 'y_test', 'train_idx', 'test_idx'
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Create target: S&P 500 monthly returns
    # Use FIN_mom_12m as proxy (12-month momentum), but compute direct return if available
    if "FIN_gspc_ret_lag1" in df.columns:
        # Create target from lagged returns - use current month's return
        target = df["FIN_gspc_ret_lag1"].shift(-1)  # Shift to align with future returns
    else:
        raise ValueError("Cannot derive target variable from available features")
    
    # Filter features to available columns
    available_economic = [f for f in ECONOMIC_FEATURES if f in df.columns]
    available_financial = [f for f in FINANCIAL_FEATURES if f in df.columns]
    
    print(f"Available economic features: {len(available_economic)}/{len(ECONOMIC_FEATURES)}")
    print(f"Available financial features: {len(available_financial)}/{len(FINANCIAL_FEATURES)}")
    
    # Create feature matrices
    X_economic = df[available_economic].copy()
    X_financial = df[available_financial].copy()
    y = target.copy()
    
    # Remove rows with any NaN values
    valid_idx = X_economic.notna().all(axis=1) & X_financial.notna().all(axis=1) & y.notna()
    
    X_economic = X_economic[valid_idx]
    X_financial = X_financial[valid_idx]
    y = y[valid_idx]
    
    print(f"\nValid samples after removing NaNs: {len(y)}")
    print(f"Target (S&P 500 returns) - Mean: {y.mean():.4f}, Std: {y.std():.4f}")
    
    # Train/test split (preserve temporal order)
    split_point = int(len(y) * train_test_split)
    train_idx = slice(0, split_point)
    test_idx = slice(split_point, len(y))
    
    return {
        'df': df,
        'X_economic': X_economic,
        'X_financial': X_financial,
        'y': y,
        'available_economic': available_economic,
        'available_financial': available_financial,
        'train_idx': train_idx,
        'test_idx': test_idx,
    }


def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names, model_name):
    """
    Train linear regression model and evaluate performance.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test feature sets
    y_train, y_test : pd.Series
        Training and test targets
    feature_names : list
        Names of features for reporting
    model_name : str
        Name of the model for printing
    
    Returns:
    --------
    dict with model, scaler, predictions, and metrics
    """
    print(f"\n{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_names)}\n")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print("PERFORMANCE METRICS")
    print("-" * 70)
    print(f"{'Metric':<25} {'Training':<20} {'Testing':<20}")
    print("-" * 70)
    print(f"{'R² Score':<25} {train_r2:>18.4f} {test_r2:>18.4f}")
    print(f"{'MSE':<25} {train_mse:>18.6f} {test_mse:>18.6f}")
    print(f"{'RMSE':<25} {train_rmse:>18.6f} {test_rmse:>18.6f}")
    print(f"{'MAE':<25} {train_mae:>18.6f} {test_mae:>18.6f}")
    print("-" * 70)
    
    # Feature coefficients
    print("\nFEATURE COEFFICIENTS (standardized)")
    print("-" * 70)
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Abs_Coef': np.abs(model.coef_)
    }).sort_values('Abs_Coef', ascending=False)
    
    for idx, row in coef_df.iterrows():
        print(f"{row['Feature']:<35} {row['Coefficient']:>12.6f}")
    
    print(f"\nIntercept: {model.intercept_:.6f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_metrics': {
            'mse': train_mse,
            'rmse': train_rmse,
            'mae': train_mae,
            'r2': train_r2,
        },
        'test_metrics': {
            'mse': test_mse,
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2,
        },
        'coef_df': coef_df,
    }


def plot_results(data, macro_results, sentiment_results, output_dir: Path = None):
    """
    Create visualization comparing model performance.
    
    Parameters:
    -----------
    data : dict
        Output from load_and_prepare_data()
    macro_results : dict
        Output from train_and_evaluate_model() for macro model
    sentiment_results : dict
        Output from train_and_evaluate_model() for sentiment model
    output_dir : Path, optional
        Directory to save figures
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract test data
    y_test = data['y'].iloc[data['test_idx']]
    y_macro_pred = macro_results['y_test_pred']
    y_sentiment_pred = sentiment_results['y_test_pred']
    test_dates = data['y'].index[data['test_idx']]
    
    # Plot 1: Economic model - Actual vs Predicted
    ax = axes[0, 0]
    ax.plot(test_dates, y_test.values, 'o-', label='Actual', linewidth=2, markersize=4)
    ax.plot(test_dates, y_macro_pred, 's--', label='Predicted', linewidth=2, markersize=4, alpha=0.7)
    ax.set_title('Economic Model: Actual vs Predicted', fontweight='bold')
    ax.set_ylabel('S&P 500 Monthly Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Financial model - Actual vs Predicted
    ax = axes[0, 1]
    ax.plot(test_dates, y_test.values, 'o-', label='Actual', linewidth=2, markersize=4)
    ax.plot(test_dates, y_sentiment_pred, 's--', label='Predicted', linewidth=2, markersize=4, alpha=0.7, color='orange')
    ax.set_title('Financial Model: Actual vs Predicted', fontweight='bold')
    ax.set_ylabel('S&P 500 Monthly Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: R² Score Comparison
    ax = axes[1, 0]
    models = ['Economic', 'Financial']
    train_r2 = [macro_results['train_metrics']['r2'], sentiment_results['train_metrics']['r2']]
    test_r2 = [macro_results['test_metrics']['r2'], sentiment_results['test_metrics']['r2']]
    
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, train_r2, width, label='Training', color='steelblue')
    ax.bar(x + width/2, test_r2, width, label='Testing', color='coral')
    ax.set_ylabel('R² Score')
    ax.set_title('Model Performance: R² Score Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: RMSE Comparison
    ax = axes[1, 1]
    train_rmse = [macro_results['train_metrics']['rmse'], sentiment_results['train_metrics']['rmse']]
    test_rmse = [macro_results['test_metrics']['rmse'], sentiment_results['test_metrics']['rmse']]
    
    x = np.arange(len(models))
    ax.bar(x - width/2, train_rmse, width, label='Training', color='steelblue')
    ax.bar(x + width/2, test_rmse, width, label='Testing', color='coral')
    ax.set_ylabel('RMSE')
    ax.set_title('Model Performance: RMSE Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "linear_regression_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison chart to {filepath}")
    else:
        plt.show()
    
    plt.close()


def plot_residuals(data, macro_results, sentiment_results, output_dir: Path = None):
    """
    Create residual plots for both models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    y_test = data['y'].iloc[data['test_idx']]
    y_macro_pred = macro_results['y_test_pred']
    y_sentiment_pred = sentiment_results['y_test_pred']
    test_dates = data['y'].index[data['test_idx']]
    
    # Economic residuals over time
    ax = axes[0, 0]
    macro_residuals = y_test.values - y_macro_pred
    ax.plot(test_dates, macro_residuals, 'o-', color='steelblue', markersize=4)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Economic Model: Residuals Over Time', fontweight='bold')
    ax.set_ylabel('Residual')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Financial residuals over time
    ax = axes[0, 1]
    sentiment_residuals = y_test.values - y_sentiment_pred
    ax.plot(test_dates, sentiment_residuals, 'o-', color='orange', markersize=4)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Financial Model: Residuals Over Time', fontweight='bold')
    ax.set_ylabel('Residual')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Economic residual distribution
    ax = axes[1, 0]
    ax.hist(macro_residuals, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Economic Model: Residual Distribution', fontweight='bold')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Financial residual distribution
    ax = axes[1, 1]
    ax.hist(sentiment_residuals, bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Financial Model: Residual Distribution', fontweight='bold')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "residual_analysis.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved residual analysis to {filepath}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Run linear regression analysis."""
    print("\n" + "="*70)
    print("LINEAR REGRESSION: ECONOMIC vs FINANCIAL INDICATORS")
    print("="*70)
    
    # Load data
    data_path = Path("data/clean/standardized_monthly_data.csv")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    data = load_and_prepare_data(data_path)
    
    # Extract train/test indices
    X_economic_train = data['X_economic'].iloc[data['train_idx']]
    X_economic_test = data['X_economic'].iloc[data['test_idx']]
    X_financial_train = data['X_financial'].iloc[data['train_idx']]
    X_financial_test = data['X_financial'].iloc[data['test_idx']]
    y_train = data['y'].iloc[data['train_idx']]
    y_test = data['y'].iloc[data['test_idx']]
    
    # Train economic model
    macro_results = train_and_evaluate_model(
        X_economic_train, X_economic_test, y_train, y_test,
        data['available_economic'],
        "MODEL 1: ECONOMIC FEATURES"
    )
    
    # Train financial model
    sentiment_results = train_and_evaluate_model(
        X_financial_train, X_financial_test, y_train, y_test,
        data['available_financial'],
        "MODEL 2: FINANCIAL FEATURES"
    )
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Metric':<30} {'Economic':<20} {'Financial':<20}")
    print("-" * 70)
    print(f"{'Test R²':<30} {macro_results['test_metrics']['r2']:>18.4f} {sentiment_results['test_metrics']['r2']:>18.4f}")
    print(f"{'Test RMSE':<30} {macro_results['test_metrics']['rmse']:>18.6f} {sentiment_results['test_metrics']['rmse']:>18.6f}")
    print(f"{'Test MAE':<30} {macro_results['test_metrics']['mae']:>18.6f} {sentiment_results['test_metrics']['mae']:>18.6f}")
    print(f"{'Train R²':<30} {macro_results['train_metrics']['r2']:>18.4f} {sentiment_results['train_metrics']['r2']:>18.4f}")
    print("-" * 70)
    
    better_model = 'Economic' if macro_results['test_metrics']['r2'] > sentiment_results['test_metrics']['r2'] else 'Financial'
    print(f"\nBetter test performance: {better_model} Model")
    print(f"Test R² advantage: {abs(macro_results['test_metrics']['r2'] - sentiment_results['test_metrics']['r2']):.4f}")
    
    # Generate visualizations
    output_dir = Path("visuals")
    plot_results(data, macro_results, sentiment_results, output_dir)
    plot_residuals(data, macro_results, sentiment_results, output_dir)
    
    print(f"\n{'='*70}")
    print("Analysis complete! Check visuals/ directory for charts.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
