import numpy as np
import pandas as pd


class FeatureEngineer:

    def __init__(self, prices, vix, usdinr):

        self.prices = prices
        self.vix = vix
        self.usdinr = usdinr

    def compute_returns(self):

        return (
            self.prices
            .pct_change()
            .dropna()
        )

    def realized_volatility(self, returns):

        return (
            returns
            .rolling(20)
            .std()
            * np.sqrt(252)
        )

    def momentum(self):

        return self.prices.pct_change(20)

    def moving_average_spread(self):

        short_ma = self.prices.rolling(20).mean()
        long_ma = self.prices.rolling(100).mean()

        return (
            short_ma - long_ma
        ) / long_ma

    def rolling_sharpe(self, returns):

        rolling_mean = (
            returns
            .rolling(60)
            .mean()
            * 252
        )

        rolling_vol = (
            returns
            .rolling(60)
            .std()
            * np.sqrt(252)
        )

        return rolling_mean / rolling_vol

    def correlation_features(self, returns):

        nifty_gold_corr = (
            returns["^NSEI"]
            .rolling(60)
            .corr(returns["GOLDBEES.NS"])
        )

        nifty_bank_corr = (
            returns["^NSEI"]
            .rolling(60)
            .corr(returns["^NSEBANK"])
        )

        return pd.DataFrame({
            "NIFTY_GOLD_CORR": nifty_gold_corr,
            "NIFTY_BANK_CORR": nifty_bank_corr
        })

    def build_feature_matrix(self):

        returns = self.compute_returns()

        realized_vol = self.realized_volatility(returns)
        momentum = self.momentum()
        ma_spread = self.moving_average_spread()
        rolling_sharpe = self.rolling_sharpe(returns)

        correlations = self.correlation_features(returns)

        features = pd.concat([

            returns,
            realized_vol,
            momentum,
            ma_spread,
            rolling_sharpe,
            correlations,

            self.vix
            .pct_change()
            .squeeze()
            .rename("INDIA_VIX"),

            self.usdinr
            .pct_change()
            .squeeze()
            .rename("USDINR")

        ], axis=1)

        features = features.dropna()

        returns = returns.loc[features.index]
        prices = self.prices.loc[features.index]

        return features, returns, prices