# Regime-Shift  
## Macro-Aware Tactical Asset Allocation Engine for Indian Markets

A quantitative finance and machine learning project that combines:

- Hidden Markov Models (HMMs)
- Regime Detection
- Tactical Asset Allocation
- Portfolio Optimization
- Walk-Forward Backtesting
- Risk Analytics

to dynamically adapt portfolio allocations across Indian financial markets.

---

# Project Objective

Traditional static portfolios fail to adapt to changing market conditions.

This project aims to:

- detect hidden market regimes,
- model volatility clustering,
- identify macro-economic transitions,
- and dynamically rebalance portfolios based on inferred latent states.

The system combines:
- machine learning,
- quantitative finance,
- and portfolio optimization

to create a macro-aware adaptive allocation framework.

---

# Key Features

## Machine Learning

- Gaussian Hidden Markov Models
- Latent regime inference
- Regime transition modeling
- State probability estimation

---

## Quantitative Finance

- Tactical Asset Allocation
- Mean-Variance Optimization
- Dynamic Risk Exposure
- Transaction Cost Modeling
- Walk-Forward Validation

---

## Risk Analytics

- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Calmar Ratio
- Rolling Volatility
- Rolling Sharpe Analysis

---

## Visualization

- Regime overlays on price data
- Transition matrix heatmaps
- Equity curves
- Drawdown analysis
- Regime probability tracking

---

# Indian Market Adaptation

The framework is adapted specifically for Indian financial markets.

## Assets

| Asset | Role |
|---|---|
| ^NSEI | NIFTY 50 Benchmark |
| ^NSEBANK | Banking Sector Exposure |
| GOLDBEES.NS | Gold Hedge |
| LIQUIDBEES.NS | Cash / Defensive Allocation |

---

## Macro Features

- India VIX
- USD/INR
- Rolling correlations
- Volatility regimes
- Momentum spreads
- Sector leadership

---

# Folder Structure

```text
regime-shift/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── notebooks/
│   ├── research.ipynb
│   ├── 01_data_pipeline.ipynb
│   ├── 02_hmm_regime_detection.ipynb
│   ├── 03_portfolio_optimization.ipynb
│   ├── 04_backtest.ipynb
│   ├── 05_performance_analytics.ipynb
│   └── 06_walk_forward_validation.ipynb
│
├── src/
│   ├── ingestion/
│   ├── features/
│   ├── models/
│   ├── optimization/
│   ├── backtest/
│   ├── analytics/
│   ├── visualization/
│   └── utils/
│
├── configs/
│   └── config.yaml
│
├── reports/
│
├── tests/
│
├── requirements.txt
│
├── main.py
│
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone <repository_url>
cd regime-shift
```

---

## Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn hmmlearn cvxpy PyPortfolioOpt yfinance quantstats scipy statsmodels pyyaml vectorbt
```

---

# Running the Project

## Run Full Pipeline

```bash
python main.py
```

---

# Running the Research Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/research.ipynb
```

---

# Machine Learning Pipeline

## 1. Data Ingestion

Market data is downloaded using:

- Yahoo Finance
- NSE tickers
- India-specific ETFs

---

## 2. Feature Engineering

Features include:

- returns
- realized volatility
- momentum
- rolling Sharpe
- moving average spreads
- correlation structures
- India VIX
- USDINR

---

## 3. Regime Detection

The system trains a:

### Gaussian Hidden Markov Model

to identify latent market states.

Example regimes:

| Regime | Interpretation |
|---|---|
| Low Vol + Strong Banks | Bull Market |
| High VIX + Weak INR | Crisis |
| Gold Rally + Falling Equities | Risk-Off |
| Recovery + Falling Vol | Recovery |

---

## 4. Dynamic Allocation

Portfolio weights adapt according to inferred regimes.

Example:

### Bull Regime

```text
80% Equities
10% Gold
10% Cash
```

### Crisis Regime

```text
20% Equities
50% Cash
30% Gold
```

---

## 5. Portfolio Optimization

Uses:

### Mean-Variance Optimization

with:

- risk aversion constraints
- long-only exposure
- covariance estimation

via:

```python
cvxpy
```

---

## 6. Backtesting

The system includes:

- walk-forward validation
- rolling retraining
- transaction cost modeling
- turnover penalties

to avoid:

- look-ahead bias
- overfitting
- unrealistic backtests

---

# Performance Metrics

The framework computes:

| Metric | Purpose |
|---|---|
| CAGR | Long-term growth |
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk efficiency |
| Max Drawdown | Tail-risk severity |
| Calmar Ratio | Return vs Drawdown |

---

# Example Visualizations

## Regime Overlay

Market prices colored by detected hidden state.

---

## Transition Matrix

Probability of regime transitions:


P(S_t = j | S_t-1 = i)


---

## Equity Curves

Compare:

- Regime Strategy
- NIFTY Benchmark

---

# Research Motivation

Financial markets exhibit:

- non-stationarity
- volatility clustering
- structural breaks
- hidden macro-economic states

Traditional static allocation frameworks fail to adapt.

This project attempts to model markets as:

## Probabilistic Regime-Switching Systems

where asset behavior changes conditional on latent states.

---

# Future Improvements

## Advanced Regime Models

- Hidden Semi-Markov Models (HSMM)
- Bayesian HMMs
- Switching Kalman Filters

---

## Portfolio Construction

- Risk Parity
- CVaR Optimization
- Black-Litterman Allocation

---

## Alternative Data

- FII/DII Flows
- RBI Policy Data
- Options Implied Volatility
- Yield Curve Dynamics

---

## Machine Learning Extensions

- Reinforcement Learning Allocation
- Online Learning
- Transformer Time-Series Models
- Regime Forecasting Networks

---

# Academic & Quant Concepts Used

## Machine Learning

- Hidden Markov Models
- Probabilistic State Inference
- Latent Variable Models

---

## Statistics

- Covariance Estimation
- Volatility Modeling
- Correlation Structures

---

## Quantitative Finance

- Tactical Asset Allocation
- Risk Management
- Portfolio Optimization
- Drawdown Analysis

---

# Recommended Next Steps

1. Add India VIX directly into feature matrix
2. Introduce sector rotation ETFs
3. Add walk-forward retraining
4. Implement probability-weighted allocation
5. Add live market deployment
6. Integrate NSE/FRED macro datasets

---

# Disclaimer

This project is for:

- educational purposes,
- research,
- and quantitative experimentation.

It is not financial advice.

---

# Author

Quantitative Finance + Machine Learning Research Project

Built using:

- Python
- HMMs
- CVXPY
- Pandas
- Quantitative Portfolio Theory
