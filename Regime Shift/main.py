import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data.data_loader import IndianMarketDataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.hmm_model import HMMRegimeModel
from src.optimization.portfolio_optimizer import RegimePortfolioOptimizer
from src.backtesting.backtester import RegimeBacktester
from src.analytics.performance_metrics import PerformanceAnalytics
from src.walkforward.walkforward import  WalkForwardRunner

from src.visualization.visualization import (
    plot_regimes,
    plot_regime_probabilities,
    plot_equity_curve,
    plot_drawdown,
    plot_turnover,
    plot_rolling_sharpe,
    plot_regime_distribution,
    plot_transition_matrix
)

from src.utils.utils import (
    create_directories,
    save_dataframe
)


# ==========================================
# CREATE FOLDERS
# ==========================================

create_directories()


# ==========================================
# CONFIGURATION
# ==========================================

TICKERS = [

    "^NSEI",
    "^NSEBANK",
    "GOLDBEES.NS",
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "INFY.NS"

]

START_DATE = "2010-01-01"
END_DATE = "2025-01-01"

N_REGIMES = 3


# ==========================================
# LOAD DATA
# ==========================================

loader = IndianMarketDataLoader(
    tickers=TICKERS,
    start_date=START_DATE,
    end_date=END_DATE
)

prices, india_vix, usdinr = (
    loader.load_all_data()
)

print("Data Loaded")


# ==========================================
# FEATURE ENGINEERING
# ==========================================

feature_engineer = FeatureEngineer(
    prices,
    india_vix,
    usdinr
)

features, returns, aligned_prices = (
    feature_engineer.build_feature_matrix()
)

print("Features Created")


# ==========================================
# SAVE DATA
# ==========================================

save_dataframe(
    prices,
    "data/processed/prices.csv"
)

save_dataframe(
    features,
    "data/processed/features.csv"
)

save_dataframe(
    returns,
    "data/processed/returns.csv"
)


# ==========================================
# WALK-FORWARD PIPELINE
# ==========================================

from src.walkforward.walkforward import (
    WalkForwardRunner
)

walkforward = WalkForwardRunner(
    features=features,
    returns=returns,
    train_window=756,
    test_window=21,
    n_regimes=N_REGIMES

)

strategy_returns, regimes, probabilities, turnover = (
    walkforward.run()
)

print("Walk-forward complete")

save_dataframe(
    regimes.to_frame(),
    "data/processed/walkforward_regimes.csv"
)

save_dataframe(
    probabilities,
    "data/processed/walkforward_probabilities.csv"
)
# ==========================================
# VISUALIZATION
# ==========================================

plot_regimes(
    prices,
    regimes
)

plot_regime_probabilities(
    probabilities
)

plot_equity_curve(

    strategy_returns,

    returns.loc[
        strategy_returns.index,
        "^NSEI"
    ]

)

plot_drawdown(
    strategy_returns
)

plot_turnover(
    turnover
)

plot_rolling_sharpe(
    strategy_returns
)

plot_regime_distribution(
    regimes
)

plot_transition_matrix(
    regimes,
    N_REGIMES
)

print("Plots Saved")
# ==========================================
# PERFORMANCE METRICS
# ==========================================

metrics = (
    PerformanceAnalytics
    .performance_summary(
        strategy_returns
    )
)

print("\nPerformance Metrics\n")

print(metrics)


# ==========================================
# SAVE FINAL OUTPUTS
# ==========================================

save_dataframe(
    strategy_returns.to_frame(),
    "data/processed/strategy_returns.csv"
)

save_dataframe(
    turnover.to_frame(),
    "data/processed/turnover.csv"
)

save_dataframe(
    metrics,
    "data/processed/performance_metrics.csv"
)

print("\nAll Outputs Saved")