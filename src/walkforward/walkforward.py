import numpy as np
import pandas as pd

from src.models.hmm_model import HMMRegimeModel
from src.optimization.portfolio_optimizer import (
    RegimePortfolioOptimizer
)


class WalkForwardRunner:

    def __init__(
        self,
        features,
        returns,
        train_window=756,
        test_window=21,
        n_regimes=3
    ):

        self.features = features
        self.returns = returns

        self.train_window = train_window
        self.test_window = test_window

        self.n_regimes = n_regimes

    def run(self):

        strategy_returns = []
        turnover_history = []

        regime_history = []

        probability_history = []

        dates = []

        for start in range(

            self.train_window,

            len(self.features) - self.test_window,

            self.test_window

        ):

            # ==================================
            # TRAIN / TEST SPLIT
            # ==================================

            train_features = self.features.iloc[
                start - self.train_window:start
            ]

            test_features = self.features.iloc[
                start:start + self.test_window
            ]

            test_returns = self.returns.iloc[
                start:start + self.test_window
            ]

            # ==================================
            # TRAIN HMM
            # ==================================

            hmm = HMMRegimeModel(
                n_regimes=self.n_regimes
            )

            train_regimes, _ = hmm.fit_predict(
                train_features
            )

            # ==================================
            # PREDICT TEST REGIMES
            # ==================================

            processed_test = (
                hmm.transform_features(
                    test_features
                )
            )

            predicted_regimes = (
                hmm.model.predict(
                    processed_test
                )
            )

            predicted_probs = (
                hmm.model.predict_proba(
                    processed_test
                )
            )

            # ==================================
            # CONFIDENCE-BASED FILTER
            # ==================================

            filtered_regimes = (
            predicted_regimes.copy()
            )

            confidence_threshold = 0.55

            for i in range(1, len(filtered_regimes)):

                current_regime = (
                    predicted_regimes[i]
                )

                previous_regime = (
                    filtered_regimes[i - 1]
                )

                regime_confidence = np.max(
                    predicted_probs[i]
                )

                # Immediate switch into worst regime
                if (current_regime != previous_regime
                    and regime_confidence < 0.55):

                    filtered_regimes[i] = (
                        previous_regime
                    )

                # Confidence filter elsewhere
                elif (
                    current_regime != previous_regime
                    and regime_confidence < confidence_threshold
                ):

                    filtered_regimes[i] = (
                        previous_regime
                    )

            # ==================================
            # REGIME-WISE OPTIMIZATION
            # ==================================

            optimizer = (
                RegimePortfolioOptimizer()
            )

            allocations = {}

            for regime in np.unique(
                train_regimes
            ):

                regime_mask = (
                    train_regimes == regime
                )

                regime_returns = (
                    self.returns.loc[
                        train_regimes.index
                    ]
                    .loc[regime_mask]
                )

                expected_returns = (
                    regime_returns.mean().values
                )

                covariance_matrix = (
                    regime_returns.cov().values
                )

                weights = optimizer.optimize(
                    expected_returns,
                    covariance_matrix
                )

                allocations[regime] = weights

            # ==================================
            # GENERATE TEST RETURNS
            # ==================================

            previous_weights = None

            for i in range(len(test_returns)):

                regime = filtered_regimes[i]

                weights = allocations[regime]

                # ==================================
                # DAILY PORTFOLIO RETURN
                # ==================================

                portfolio_vol = np.sqrt(

                    weights.T
                    @ covariance_matrix
                    @ weights

                ) * np.sqrt(252)

                target_vol = 0.18

                vol_scalar = min(
                    1.25,
                    target_vol / portfolio_vol
                )

                scaled_weights = (
                    weights * vol_scalar
                )

                daily_return = np.dot(

                    test_returns.iloc[i],

                    scaled_weights

                )

                # ==================================
                # TURNOVER CALCULATION
                # ==================================

                if previous_weights is not None:

                    turnover = np.abs(

                        weights - previous_weights

                    ).sum()

                else:

                    turnover = 0

                turnover_history.append(
                    turnover
                )

                previous_weights = weights

                # ==================================
                # STORE RESULTS
                # ==================================

                strategy_returns.append(
                    daily_return
                )

                regime_history.append(
                    regime
                )

                probability_history.append(
                    predicted_probs[i]
                )

                dates.append(
                    test_returns.index[i]
                )
                if previous_weights is not None:

                    turnover = np.abs(
                        weights - previous_weights
                    ).sum()

                else:

                        turnover = 0
                        turnover_history.append(turnover)
                        previous_weights = weights

        # ======================================
        # FINAL OUTPUTS
        # ======================================

        strategy_returns = pd.Series(
            strategy_returns,
            index=dates,
            name="Strategy"
        )

        regimes = pd.Series(
            regime_history,
            index=dates,
            name="Regime"
        )

        probability_df = pd.DataFrame(
            probability_history,
            index=dates
        )

        probability_df.columns = [
            f"Regime_{i}"
            for i in range(self.n_regimes)
        ]

        turnover_series = pd.Series(
        turnover_history,
        index=dates,
        name="Turnover")

        return (
            strategy_returns,
            regimes,
            probability_df,
            turnover_series
        )