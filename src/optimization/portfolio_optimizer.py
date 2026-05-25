import cvxpy as cp
import numpy as np
import pandas as pd


class RegimePortfolioOptimizer:

    def __init__(self, risk_aversion=0.5):

        self.risk_aversion = risk_aversion

    def optimize(self, expected_returns, covariance_matrix):

        n_assets = len(expected_returns)

        weights = cp.Variable(n_assets)

        portfolio_return = (
            expected_returns @ weights
        )

        portfolio_risk = cp.quad_form(
            weights,
            covariance_matrix
        )

        objective = cp.Maximize(
            portfolio_return
            - self.risk_aversion * portfolio_risk
        )

        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]

        problem = cp.Problem(
            objective,
            constraints
        )

        problem.solve()

        return np.array(
            weights.value
        ).flatten()

    def save_allocations(
        self,
        allocations,
        asset_names
    ):

        allocation_df = pd.DataFrame(
            allocations,
            index=asset_names
        )

        allocation_df.to_csv(
            "data/processed/optimized_allocations.csv"
        )

        print(
            "Allocations saved."
        )