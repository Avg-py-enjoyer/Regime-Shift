import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RegimeBacktester:

    def __init__(
        self,
        returns,
        regimes,
        allocations,
        transaction_cost=0.002
    ):

        self.returns = returns
        self.regimes = regimes
        self.allocations = allocations

        self.transaction_cost = transaction_cost

    def run_backtest(self):

        strategy_returns = []
        turnover_history = []

        previous_weights = None

        for date in self.returns.index:

            regime = self.regimes.loc[date]

            weights = self.allocations[
                regime
            ]

            portfolio_return = np.dot(
                self.returns.loc[date],
                weights
            )

            if previous_weights is not None:

                turnover = np.abs(
                    weights - previous_weights
                ).sum()

                cost = (
                    turnover
                    * self.transaction_cost
                )

            else:

                turnover = 0
                cost = 0

            portfolio_return -= cost

            strategy_returns.append(
                portfolio_return
            )

            turnover_history.append(
                turnover
            )

            previous_weights = weights

        strategy_returns = pd.Series(
            strategy_returns,
            index=self.returns.index,
            name="Strategy"
        )

        turnover_series = pd.Series(
            turnover_history,
            index=self.returns.index,
            name="Turnover"
        )

        # ======================================
        # EQUITY CURVES
        # ======================================

        strategy_curve = (
            1 + strategy_returns
        ).cumprod()

        benchmark_curve = (
            1 + self.returns["^NSEI"]
        ).cumprod()

        # ======================================
        # SAVE CSV FILES
        # ======================================

        strategy_returns.to_csv(
            "data/processed/strategy_returns.csv"
        )

        strategy_curve.to_csv(
            "data/processed/strategy_equity_curve.csv"
        )

        benchmark_curve.to_csv(
            "data/processed/benchmark_equity_curve.csv"
        )

        turnover_series.to_csv(
            "data/processed/turnover.csv"
        )

        # ======================================
        # EQUITY CURVE PLOT
        # ======================================

        plt.figure(figsize=(15, 7))

        plt.plot(
            strategy_curve,
            label="Regime Strategy"
        )

        plt.plot(
            benchmark_curve,
            label="NIFTY Benchmark"
        )

        plt.legend()

        plt.title(
            "Regime Strategy vs NIFTY"
        )

        plt.xlabel("Date")

        plt.ylabel("Growth of ₹1")

        plt.grid(True)

        plt.savefig(
            "reports/equity_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        # ======================================
        # TURNOVER PLOT
        # ======================================

        plt.figure(figsize=(15, 5))

        plt.plot(
            turnover_series
        )

        plt.title(
            "Portfolio Turnover"
        )

        plt.xlabel("Date")

        plt.ylabel("Turnover")

        plt.grid(True)

        plt.savefig(
            "reports/turnover_plot.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        return strategy_returns, turnover_series