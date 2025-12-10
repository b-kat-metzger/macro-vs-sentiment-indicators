"""
Ensemble regression models (Bagging, Boosting, Random Forest) comparing 
economic vs financial indicators. Predicts S&P 500 monthly returns using 
two separate feature sets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import (
    RandomForestRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
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
    dict with keys: 'df', 'X_economic', 'X_financial', 'y', 'available_economic',
                    'available_financial', 'train_idx', 'test_idx'
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Create target: S&P 500 monthly returns
    if "FIN_gspc_ret_lag1" in df.columns:
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


def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names, model_name, 
                            model_type='random_forest'):
    """
    Train ensemble regression model and evaluate performance.
    
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
    model_type : str
        Type of ensemble model: 'random_forest', 'bagging', 'gradient_boosting', 'adaboost'
    
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
    
    # Train model based on type
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=200, max_depth=15, 
                                     min_samples_split=5, min_samples_leaf=2,
                                     random_state=42, n_jobs=-1)
    elif model_type == 'bagging':
        base_estimator = DecisionTreeRegressor(max_depth=15)
        model = BaggingRegressor(estimator=base_estimator, n_estimators=200,
                                random_state=42, n_jobs=-1)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                         max_depth=5, min_samples_split=5,
                                         min_samples_leaf=2, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
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
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print("\nFEATURE IMPORTANCE (Top 10)")
        print("-" * 70)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for idx, row in importance_df.head(10).iterrows():
            print(f"{row['Feature']:<35} {row['Importance']:>12.6f}")
    
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
    }


def plot_aggregate_results(data, all_models_results, output_dir: Path = None):
    """
    Create aggregate visualization showing all three models' economic and financial features.
    
    Parameters:
    -----------
    data : dict
        Output from load_and_prepare_data()
    all_models_results : dict
        Dictionary with structure {model_name: {economic: results, financial: results}}
    output_dir : Path, optional
        Directory to save figures
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    # Extract test data (same for all models)
    y_test = data['y'].iloc[data['test_idx']]
    test_dates = data['y'].index[data['test_idx']]
    
    model_names = ['Random Forest', 'Bagging', 'Gradient Boosting']
    colors_econ = ['#1f77b4', '#ff7f0e']  # Blue for actual, orange for predicted
    
    for row_idx, model_name in enumerate(model_names):
        results = all_models_results[model_name]
        
        # Economic model (left column)
        ax = axes[row_idx, 0]
        y_econ_pred = results['economic']['y_test_pred']
        econ_r2 = results['economic']['test_metrics']['r2']
        
        ax.plot(test_dates, y_test.values, 'o-', label='Actual', linewidth=2, markersize=3, color='#1f77b4')
        ax.plot(test_dates, y_econ_pred, 's--', label='Predicted', linewidth=2, markersize=3, alpha=0.7, color='#ff7f0e')
        ax.set_title(f'{model_name}: Economic Features (R² = {econ_r2:.4f})', fontweight='bold', fontsize=11)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('S&P 500 Monthly Return', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
        ax.tick_params(axis='y', labelsize=9)
        
        # Financial model (right column)
        ax = axes[row_idx, 1]
        y_fin_pred = results['financial']['y_test_pred']
        fin_r2 = results['financial']['test_metrics']['r2']
        
        ax.plot(test_dates, y_test.values, 'o-', label='Actual', linewidth=2, markersize=3, color='#1f77b4')
        ax.plot(test_dates, y_fin_pred, 's--', label='Predicted', linewidth=2, markersize=3, alpha=0.7, color='#2ca02c')
        ax.set_title(f'{model_name}: Financial Features (R² = {fin_r2:.4f})', fontweight='bold', fontsize=11)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('S&P 500 Monthly Return', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
        ax.tick_params(axis='y', labelsize=9)
    
    fig.suptitle('Ensemble Models Comparison: Economic vs Financial Features', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "ensemble_aggregate_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nSaved aggregate comparison chart to {filepath}")
    else:
        plt.show()
    
    plt.close()


def plot_results(data, economic_results, financial_results, model_type, output_dir: Path = None):
    """
    Create visualization comparing model performance.
    
    Parameters:
    -----------
    data : dict
        Output from load_and_prepare_data()
    economic_results : dict
        Output from train_and_evaluate_model() for economic model
    financial_results : dict
        Output from train_and_evaluate_model() for financial model
    model_type : str
        Type of model for title
    output_dir : Path, optional
        Directory to save figures
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract test data
    y_test = data['y'].iloc[data['test_idx']]
    y_econ_pred = economic_results['y_test_pred']
    y_fin_pred = financial_results['y_test_pred']
    test_dates = data['y'].index[data['test_idx']]
    
    # Plot 1: Economic model - Actual vs Predicted
    ax = axes[0]
    ax.plot(test_dates, y_test.values, 'o-', label='Actual', linewidth=2, markersize=4)
    ax.plot(test_dates, y_econ_pred, 's--', label='Predicted', linewidth=2, markersize=4, alpha=0.7)
    ax.set_title(f'{model_type}: Economic Features - Actual vs Predicted', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('S&P 500 Monthly Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Financial model - Actual vs Predicted
    ax = axes[1]
    ax.plot(test_dates, y_test.values, 'o-', label='Actual', linewidth=2, markersize=4)
    ax.plot(test_dates, y_fin_pred, 's--', label='Predicted', linewidth=2, markersize=4, alpha=0.7, color='orange')
    ax.set_title(f'{model_type}: Financial Features - Actual vs Predicted', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('S&P 500 Monthly Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{model_type.lower()}_eco_fin.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def main():
    """Run ensemble regression analysis for all model types."""
    print("\n" + "="*70)
    print("ENSEMBLE REGRESSION: ECONOMIC vs FINANCIAL INDICATORS")
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
    
    # Model types to train
    model_types = [
        ('Random Forest', 'random_forest'),
        ('Bagging', 'bagging'),
        ('Gradient Boosting', 'gradient_boosting'),
    ]
    
    results_summary = {}
    all_models_data = {}  # Store all model results for aggregate visualization
    
    for display_name, model_type in model_types:
        print(f"\n\n{'#'*70}")
        print(f"# {display_name.upper()}")
        print(f"{'#'*70}")
        
        # Train economic model
        econ_results = train_and_evaluate_model(
            X_economic_train, X_economic_test, y_train, y_test,
            data['available_economic'],
            f"MODEL 1: ECONOMIC FEATURES ({display_name})",
            model_type=model_type
        )
        
        # Train financial model
        fin_results = train_and_evaluate_model(
            X_financial_train, X_financial_test, y_train, y_test,
            data['available_financial'],
            f"MODEL 2: FINANCIAL FEATURES ({display_name})",
            model_type=model_type
        )
        
        # Summary comparison
        print(f"\n{'='*70}")
        print(f"SUMMARY: {display_name}")
        print(f"{'='*70}")
        print(f"\n{'Metric':<30} {'Economic':<20} {'Financial':<20}")
        print("-" * 70)
        print(f"{'Test R²':<30} {econ_results['test_metrics']['r2']:>18.4f} {fin_results['test_metrics']['r2']:>18.4f}")
        print(f"{'Test RMSE':<30} {econ_results['test_metrics']['rmse']:>18.6f} {fin_results['test_metrics']['rmse']:>18.6f}")
        print(f"{'Test MAE':<30} {econ_results['test_metrics']['mae']:>18.6f} {fin_results['test_metrics']['mae']:>18.6f}")
        print(f"{'Train R²':<30} {econ_results['train_metrics']['r2']:>18.4f} {fin_results['train_metrics']['r2']:>18.4f}")
        print("-" * 70)
        
        better_model = 'Economic' if econ_results['test_metrics']['r2'] > fin_results['test_metrics']['r2'] else 'Financial'
        r2_diff = abs(econ_results['test_metrics']['r2'] - fin_results['test_metrics']['r2'])
        print(f"\nBetter test performance: {better_model} Model")
        print(f"Test R² advantage: {r2_diff:.4f}")
        
        # Store results
        results_summary[display_name] = {
            'economic': econ_results['test_metrics'],
            'financial': fin_results['test_metrics'],
        }
        
        # Store for aggregate visualization
        all_models_data[display_name] = {
            'economic': {
                'y_test_pred': econ_results['y_test_pred'],
                'test_metrics': econ_results['test_metrics'],
            },
            'financial': {
                'y_test_pred': fin_results['y_test_pred'],
                'test_metrics': fin_results['test_metrics'],
            },
        }
        
        # Generate visualizations
        output_dir = Path("visuals")
        plot_results(data, econ_results, fin_results, display_name, output_dir)
    
    # Final comparison across all models
    print(f"\n\n{'#'*70}")
    print("# CROSS-MODEL COMPARISON")
    print(f"{'#'*70}")
    print(f"\n{'Model':<20} {'Economic':<20} {'Financial':<20}")
    print("-" * 70)
    for model_name, metrics in results_summary.items():
        print(f"{model_name + ' R²':<20} {metrics['economic']['r2']:>18.4f} {metrics['financial']['r2']:>18.4f}")
        print(f"{model_name + ' RMSE':<20} {metrics['economic']['rmse']:>18.6f} {metrics['financial']['rmse']:>18.6f}")
        print(f"{model_name + ' MAE':<20} {metrics['economic']['mae']:>18.6f} {metrics['financial']['mae']:>18.6f}")
        print("-" * 70)
    
    # Generate aggregate visualization for report
    output_dir = Path("visuals")
    plot_aggregate_results(data, all_models_data, output_dir)


if __name__ == "__main__":
    main()
