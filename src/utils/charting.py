"""
Visualization utilities for economic vs financial indicators analysis.
Creates publication-quality charts for economic and financial features.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple


# Feature groupings
ECO_FEATURES = [
    "ECO_dCPIAUCSL",
    "ECO_dFEDFUNDS",
    "ECO_dINDPRO",
    "FRED_AMTMNO",
    "FRED_PERMIT",
    "FRED_UNRATE",
    "FRED_ICSA",
    "FRED_UMCSENT",
    "FRED_RSAFS",
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


def load_data(filepath: Path) -> pd.DataFrame:
    """Load standardized monthly data from CSV."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a series to zero mean and unit variance."""
    return (series - series.mean()) / series.std()


def plot_eco_features(df: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
    """
    Create a multi-panel visualization of macroeconomic indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Standardized monthly data
    output_dir : Path, optional
        Directory to save figures. If None, displays interactively.
    """
    # Filter to available macro features
    available_macro = [f for f in ECO_FEATURES if f in df.columns]
    
    if not available_macro:
        print("No macroeconomic features found in data.")
        return
    
    # Create 2x3 subplots
    n_features = len(available_macro)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(available_macro):
        ax = axes[idx]
        data = df[feature].dropna()
        
        ax.plot(data.index, data.values, linewidth=1.5, color='steelblue')
        ax.fill_between(data.index, data.values, alpha=0.3, color='steelblue')
        ax.set_title(feature, fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Hide unused subplots
    for idx in range(len(available_macro), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "eco_indicators.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved economic indicators chart to {filepath}")
    else:
        plt.show()
    
    plt.close()


def plot_financial_features(df: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
    """
    Create a multi-panel visualization of financial indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Standardized monthly data
    output_dir : Path, optional
        Directory to save figures. If None, displays interactively.
    """
    # Filter to available financial features
    available_financial = [f for f in FINANCIAL_FEATURES if f in df.columns]
    
    if not available_financial:
        print("No financial features found in data.")
        return
    
    # Create 2x4 subplots
    n_features = len(available_financial)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(available_financial):
        ax = axes[idx]
        data = df[feature].dropna()
        
        # Use different color for financial indicators
        ax.plot(data.index, data.values, linewidth=1.5, color='darkred')
        ax.fill_between(data.index, data.values, alpha=0.3, color='darkred')
        ax.set_title(feature, fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Hide unused subplots
    for idx in range(len(available_financial), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "financial_indicators.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved financial indicators chart to {filepath}")
    else:
        plt.show()
    
    plt.close()


def plot_normalized_comparison(df: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
    """
    Plot normalized macroeconomic vs financial indicators on same scale for comparison.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Standardized monthly data
    output_dir : Path, optional
        Directory to save figures. If None, displays interactively.
    """
    available_macro = [f for f in ECO_FEATURES if f in df.columns]
    available_financial = [f for f in FINANCIAL_FEATURES if f in df.columns]
    
    if not available_macro or not available_financial:
        print("Need both macro and financial features for comparison.")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Normalize and plot macro
    for feature in available_macro:
        normalized = normalize_series(df[feature])
        ax1.plot(normalized.index, normalized.values, label=feature, linewidth=1, alpha=0.7)
    
    ax1.set_title('Normalized Economic Indicators', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Normalized Value (Z-score)')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Normalize and plot financial indicators
    for feature in available_financial:
        normalized = normalize_series(df[feature])
        ax2.plot(normalized.index, normalized.values, label=feature, linewidth=1, alpha=0.7)

    ax2.set_title('Normalized Financial Indicators', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Normalized Value (Z-score)')
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "normalized_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved normalized comparison chart to {filepath}")
    else:
        plt.show()
    
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
    """
    Create a correlation heatmap of macro and financial features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Standardized monthly data
    output_dir : Path, optional
        Directory to save figures. If None, displays interactively.
    """
    available_macro = [f for f in ECO_FEATURES if f in df.columns]
    available_financial = [f for f in FINANCIAL_FEATURES if f in df.columns]
    all_features = available_macro + available_financial
    
    if len(all_features) < 2:
        print("Insufficient features for correlation analysis.")
        return
    
    # Compute correlation matrix
    corr_matrix = df[all_features].corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(all_features)))
    ax.set_yticks(np.arange(len(all_features)))
    ax.set_xticklabels(all_features, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(all_features, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(all_features)):
        for j in range(len(all_features)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    # Add visual separation between macro and financial features
    ax.axhline(len(available_macro) - 0.5, color='black', linewidth=2)
    ax.axvline(len(available_macro) - 0.5, color='black', linewidth=2)
    
    ax.set_title('Feature Correlation Matrix: Economic vs Financial Indicators', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "correlation_heatmap.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved correlation heatmap to {filepath}")
    else:
        plt.show()
    
    plt.close()


def plot_summary_statistics(df: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
    """
    Create a summary statistics visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Standardized monthly data
    output_dir : Path, optional
        Directory to save figures. If None, displays interactively.
    """
    available_macro = [f for f in ECO_FEATURES if f in df.columns]
    available_financial = [f for f in FINANCIAL_FEATURES if f in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean values
    ax = axes[0, 0]
    means_macro = df[available_macro].mean()
    means_financial = df[available_financial].mean()
    all_means = pd.concat([means_macro, means_financial])
    colors = ['steelblue'] * len(available_macro) + ['darkred'] * len(available_financial)
    all_means.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Mean Values', fontweight='bold')
    ax.set_xlabel('Mean')
    
    # Standard deviation
    ax = axes[0, 1]
    std_macro = df[available_macro].std()
    std_financial = df[available_financial].std()
    all_std = pd.concat([std_macro, std_financial])
    all_std.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Standard Deviation', fontweight='bold')
    ax.set_xlabel('Std Dev')
    
    # Count of non-null values
    ax = axes[1, 0]
    counts_macro = df[available_macro].notna().sum()
    counts_financial = df[available_financial].notna().sum()
    all_counts = pd.concat([counts_macro, counts_financial])
    all_counts.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Non-Null Observations', fontweight='bold')
    ax.set_xlabel('Count')
    
    # Data availability over time
    ax = axes[1, 1]
    availability_macro = df[available_macro].notna().sum(axis=1)
    availability_financial = df[available_financial].notna().sum(axis=1)
    ax.plot(availability_macro.index, availability_macro.values, label='Macro Features', linewidth=2, color='steelblue')
    ax.plot(availability_financial.index, availability_financial.values, label='Financial Features', linewidth=2, color='darkred')
    ax.set_title('Feature Availability Over Time', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Non-Null Features')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "summary_statistics.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved summary statistics chart to {filepath}")
    else:
        plt.show()
    
    plt.close()


def generate_all_visualizations(data_path: Path, output_dir: Optional[Path] = None) -> None:
    """
    Generate all available visualizations for the dataset.
    
    Parameters:
    -----------
    data_path : Path
        Path to standardized_monthly_data.csv
    output_dir : Path, optional
        Directory to save figures. Defaults to visuals/
    """
    if output_dir is None:
        output_dir = Path("visuals")
    
    print(f"Loading data from {data_path}")
    df = load_data(data_path)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print()
    
    print("Generating visualizations...")
    plot_eco_features(df, output_dir)
    plot_financial_features(df, output_dir)
    plot_normalized_comparison(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_summary_statistics(df, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    data_path = Path("data/clean/standardized_monthly_data.csv")
    output_dir = Path("visuals")
    
    if data_path.exists():
        generate_all_visualizations(data_path, output_dir)
    else:
        print(f"Data file not found: {data_path}")
