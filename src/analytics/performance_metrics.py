import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PerformanceAnalytics:

    @staticmethod
    def CAGR(returns):

        cumulative = (
            1 + returns
        ).prod()

        years = len(returns) / 252

        return cumulative ** (
            1 / years
        ) - 1

    @staticmethod
    def sharpe_ratio(returns):

        return (
            np.sqrt(252)
            * returns.mean()
            / returns.std()
        )

    @staticmethod
    def sortino_ratio(returns):

        downside = returns[
            returns < 0
        ]

        downside_std = downside.std()

        return (
            np.sqrt(252)
            * returns.mean()
            / downside_std
        )

    @staticmethod
    def max_drawdown(returns):

        cumulative = (
            1 + returns
        ).cumprod()

        peak = cumulative.cummax()

        drawdown = (
            cumulative / peak
        ) - 1

        return drawdown.min()

    @staticmethod
    def calmar_ratio(returns):

        return (
            PerformanceAnalytics.CAGR(returns)
            /
            abs(
                PerformanceAnalytics.max_drawdown(
                    returns
                )
            )
        )

    @staticmethod
    def performance_summary(returns):

        metrics = {

            "CAGR": [
                PerformanceAnalytics.CAGR(
                    returns
                )
            ],

            "Sharpe": [
                PerformanceAnalytics.sharpe_ratio(
                    returns
                )
            ],

            "Sortino": [
                PerformanceAnalytics.sortino_ratio(
                    returns
                )
            ],

            "Max Drawdown": [
                PerformanceAnalytics.max_drawdown(
                    returns
                )
            ],

            "Calmar": [
                PerformanceAnalytics.calmar_ratio(
                    returns
                )
            ]

        }

        metrics_df = pd.DataFrame(
            metrics
        )

        # ======================================
        # SAVE METRICS
        # ======================================

        metrics_df.to_csv(
            "data/processed/performance_metrics.csv",
            index=False
        )

        # ======================================
        # DRAWDOWN PLOT
        # ======================================

        cumulative = (
            1 + returns
        ).cumprod()

        peak = cumulative.cummax()

        drawdown = (
            cumulative / peak
        ) - 1

        plt.figure(figsize=(15, 5))

        plt.plot(drawdown)

        plt.title(
            "Strategy Drawdown"
        )

        plt.xlabel("Date")

        plt.ylabel("Drawdown")

        plt.grid(True)

        plt.savefig(
            "reports/drawdown_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        return metrics_df