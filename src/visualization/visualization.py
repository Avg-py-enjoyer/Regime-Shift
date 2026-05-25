import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.utils import save_plot


# ==========================================
# REGIME DETECTION PLOT
# ==========================================

def plot_regimes(
    prices,
    regimes
):

    aligned_prices = prices.loc[
        regimes.index
    ]

    plt.figure(figsize=(16, 8))

    for regime in np.unique(regimes):

        mask = (
            regimes == regime
        )

        plt.scatter(

            aligned_prices.index[mask],

            aligned_prices["^NSEI"][mask],

            label=f"Regime {regime}",

            s=10

        )

    plt.legend()

    plt.title(
        "Walk-Forward HMM Regime Detection"
    )

    plt.xlabel("Date")

    plt.ylabel("NIFTY Index")

    plt.grid(True)

    save_plot(
    "regime_detection.png")

    plt.show()


# ==========================================
# REGIME PROBABILITY PLOT
# ==========================================

def plot_regime_probabilities(
    probabilities
):

    plt.figure(figsize=(16, 7))

    for column in probabilities.columns:

        plt.plot(

            probabilities.index,

            probabilities[column],

            label=column

        )

    plt.legend()

    plt.title(
        "Regime Probabilities Through Time"
    )

    plt.xlabel("Date")

    plt.ylabel("Probability")

    plt.grid(True)

    save_plot(
    "regime_probabilities.png"
    )

    plt.show()


# ==========================================
# EQUITY CURVE
# ==========================================

def plot_equity_curve(
    strategy_returns,
    benchmark_returns
):

    strategy_curve = (
        1 + strategy_returns
    ).cumprod()

    benchmark_curve = (
        1 + benchmark_returns
    ).cumprod()

    plt.figure(figsize=(16, 8))

    plt.plot(

        strategy_curve,

        label="Walk-Forward Strategy"

    )

    plt.plot(

        benchmark_curve,

        label="NIFTY Benchmark"

    )

    plt.legend()

    plt.title(
        "Strategy vs NIFTY"
    )

    plt.xlabel("Date")

    plt.ylabel("Growth of ₹1")

    plt.grid(True)

    save_plot(
    "equity_curve.png")

    plt.show()


# ==========================================
# DRAWDOWN PLOT
# ==========================================

def plot_drawdown(
    strategy_returns
):

    cumulative = (
        1 + strategy_returns
    ).cumprod()

    peak = cumulative.cummax()

    drawdown = (
        cumulative / peak
    ) - 1

    plt.figure(figsize=(16, 6))

    plt.plot(drawdown)

    plt.title(
        "Strategy Drawdown"
    )

    plt.xlabel("Date")

    plt.ylabel("Drawdown")

    plt.grid(True)

    save_plot(
    "drawdown_curve.png")

    plt.show()


# ==========================================
# TURNOVER PLOT
# ==========================================

def plot_turnover(
    turnover
):

    plt.figure(figsize=(16, 5))

    plt.plot(turnover)

    plt.title(
        "Portfolio Turnover"
    )

    plt.xlabel("Date")

    plt.ylabel("Turnover")

    plt.grid(True)

    save_plot(
    "turnover_plot.png")

    plt.show()


# ==========================================
# ROLLING SHARPE
# ==========================================

def plot_rolling_sharpe(
    strategy_returns,
    window=126
):

    rolling_sharpe = (

        strategy_returns
        .rolling(window)
        .mean()

        /

        strategy_returns
        .rolling(window)
        .std()

    ) * np.sqrt(252)

    plt.figure(figsize=(16, 6))

    plt.plot(
        rolling_sharpe
    )

    plt.axhline(
        1.0,
        linestyle="--"
    )

    plt.title(
        "Rolling Sharpe Ratio"
    )

    plt.xlabel("Date")

    plt.ylabel("Sharpe")

    plt.grid(True)

    save_plot(
    "rolling_sharpe.png")

    plt.show()


# ==========================================
# REGIME DISTRIBUTION
# ==========================================

def plot_regime_distribution(
    regimes
):

    plt.figure(figsize=(8, 5))

    sns.countplot(
        x=regimes
    )

    plt.title(
        "Regime Frequency Distribution"
    )

    plt.xlabel("Regime")

    plt.ylabel("Count")

    save_plot(
    "regime_distribution.png")

    plt.show()


# ==========================================
# REGIME TRANSITION HEATMAP
# ==========================================

def plot_transition_matrix(
    regimes,
    n_regimes
):

    transition_matrix = np.zeros(
        (n_regimes, n_regimes)
    )

    for i in range(len(regimes) - 1):

        current_regime = regimes.iloc[i]

        next_regime = regimes.iloc[i + 1]

        transition_matrix[
            current_regime,
            next_regime
        ] += 1

    row_sums = (
        transition_matrix.sum(axis=1)
        .reshape(-1, 1)
    )

    transition_matrix = (
        transition_matrix / row_sums
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(

        transition_matrix,

        annot=True,

        cmap="viridis"

    )

    plt.title(
        "Empirical Transition Matrix"
    )

    plt.xlabel("Next Regime")

    plt.ylabel("Current Regime")

    save_plot(
    "transition_matrix.png")

    plt.show()