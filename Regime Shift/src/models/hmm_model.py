import pandas as pd

from hmmlearn.hmm import GaussianHMM

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class HMMRegimeModel:

    def __init__(
        self,
        n_regimes=3,
        covariance_type="diag",
        n_iter=3000,
        random_state=42
    ):

        self.n_regimes = n_regimes

        self.scaler = StandardScaler()

        self.pca = PCA(
            n_components=0.95
        )

        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            min_covar=1e-4,
            random_state=random_state
        )

    def fit_transform_features(self, features):

        clipped = features.clip(
            lower=features.quantile(0.01),
            upper=features.quantile(0.99),
            axis=1
        )

        scaled = self.scaler.fit_transform(
            clipped
        )

        reduced = self.pca.fit_transform(
            scaled
        )

        return reduced


    def transform_features(self, features):

        clipped = features.clip(
            lower=features.quantile(0.01),
            upper=features.quantile(0.99),
            axis=1
        )

        scaled = self.scaler.transform(
            clipped
        )

        reduced = self.pca.transform(
            scaled
        )

        return reduced

    def fit_predict(self, features):

        processed = self.fit_transform_features(
            features
        )

        self.model.fit(processed)

        hidden_states = self.model.predict(
            processed
        )

        probabilities = self.model.predict_proba(
            processed
        )

        regimes = pd.Series(
            hidden_states,
            index=features.index,
            name="Regime"
        )

        probability_df = pd.DataFrame(
            probabilities,
            index=features.index
        )

        probability_df.columns = [
            f"Regime_{i}"
            for i in range(self.n_regimes)
        ]

        # ======================================
        # SAVE OUTPUTS
        # ======================================

        regimes.to_csv(
            "data/processed/regimes.csv"
        )

        probability_df.to_csv(
            "data/processed/regime_probabilities.csv"
        )

        return regimes, probability_df

    def transition_matrix(self):

        matrix = pd.DataFrame(
            self.model.transmat_
        )

        matrix.columns = [
            f"Regime_{i}"
            for i in range(self.n_regimes)
        ]

        matrix.index = [
            f"Regime_{i}"
            for i in range(self.n_regimes)
        ]

        matrix.to_csv(
            "data/processed/transition_matrix.csv"
        )

        return matrix