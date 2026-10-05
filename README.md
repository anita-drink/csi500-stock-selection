## Project Overview

This project developed a machine-learning pipeline for **short-horizon stock selection within the CSI500 universe** as part of the Spring 2026 CSCI-SHU 360 Machine Learning competition. The objective was to construct a valid long-only portfolio capable of outperforming the CSI500 benchmark over short live evaluation windows.

Starting from the competition-provided XGBoost baseline, I iteratively developed a more robust **LightGBM-based stock-ranking and portfolio-construction pipeline**. The final workflow combines cross-sectional return prediction, engineered momentum and risk features, recency-weighted training, regime-aware signals, model ensembling, and portfolio-level hyperparameter optimization.

Rather than optimizing only predictive metrics such as Rank IC, the project ultimately treated **portfolio construction itself as part of the machine-learning problem**, selecting models based on realized excess return against the CSI500 benchmark.

## Methodology

### Target Construction

The model predicts the **cross-sectional z-score of 5-day forward returns** rather than raw future returns.

For stock *i* at time *t*:
<p align="center">
  <b>r<sub>i,t</sub><sup>(5)</sup> = P<sub>i,t+5</sub> / P<sub>i,t</sub> - 1</b>
</p>

The returns are then standardized within each trading date:

<p align="center">
  <b>z<sub>i,t</sub> = (r<sub>i,t</sub><sup>(5)</sup> - μ<sub>t</sub>) / σ<sub>t</sub></b>
</p>

where:

- **r<sub>i,t</sub><sup>(5)</sup>** = 5-day forward return for stock *i* at time *t*
- **μ<sub>t</sub>** = mean 5-day forward return across all stocks on date *t*
- **σ<sub>t</sub>** = standard deviation of 5-day forward returns across all stocks on date *t*


This reframes the problem from predicting whether the market will rise or fall to predicting **which CSI500 stocks are likely to outperform their peers**, which better matches the benchmark-relative competition objective.

### Feature Engineering

The LightGBM models use a broad collection of market-derived features designed to capture multiple weak but complementary signals:

- Multi-horizon momentum: 3-, 5-, 10-, 20-, and 60-day returns
- Skip-one momentum
- Volatility-adjusted momentum
- Short-term reversal
- Realized and range volatility
- Drawdowns and distance from 52-week highs
- Volume and liquidity trends
- Turnover abnormality
- Amihud-style illiquidity
- Suspension / stale-trading proxies
- Moving-average distance
- RSI, overnight gaps, and intraday returns
- CSI500-relative alpha
- Market-relative volatility
- Rolling beta and idiosyncratic returns
- Market-regime interactions

The model therefore does not rely on a single trading factor; instead, LightGBM learns nonlinear interactions among momentum, volatility, liquidity, market-relative behavior, and changing market regimes.

### Model Training

The strongest model family used **LightGBM regression with Huber loss** on winsorized forward-return z-scores.

Several design decisions were introduced to improve robustness:

- **Huber loss** reduces sensitivity to extreme short-horizon returns.
- **Label winsorization at ±3 standard deviations** prevents extreme price moves from dominating training.
- **Exponential recency weighting** gives greater importance to recent market conditions.
- A **180-day rolling training window** balances sample size against regime staleness.
- A **5-day embargo** prevents overlap between forward-return labels and validation periods.
- Predictions are averaged across **five LightGBM models using seeds 42–46** to reduce stochastic model variance.

### No-Leakage Validation

All historical experiments used time-aware evaluation.

For each backtest:

1. Features and labels were restricted to information available before the historical `as-of` date.
2. A portfolio was generated using only past data.
3. The portfolio was evaluated exclusively on later trading dates.
4. Training and validation labels were additionally capped to prevent 5-day forward-return labels from crossing the historical portfolio-construction date.

This avoids look-ahead bias and ensures that the reported performance could have been produced in real time.

## Portfolio Optimization

Because the competition score depended on **realized portfolio excess return**, portfolio construction was optimized alongside the predictive model.

I ran a **240-experiment portfolio sweep** across:

- `top_k ∈ {30, 40, 50, 60, 80}`
- `objective ∈ {Huber, LambdaRank}`
- `weight_method ∈ {rank, equal, volatility-adjusted}`
- `filter_mode ∈ {none, liquidity/risk filter}`
- 4 historical no-leakage evaluation windows

Each generated portfolio was validated against the competition constraints and then evaluated using realized excess return against the CSI500.

### Main Findings

The experiments produced several consistent findings:

- **Huber regression substantially outperformed LambdaRank.**
- **Top-30 and Top-40 portfolios generally performed best**, suggesting that useful predictive signal was concentrated near the top of the ranking.
- **Rank weighting outperformed equal and volatility-adjusted weighting**, indicating that the model's ordering among selected stocks contained useful information.
- Liquidity/risk filtering slightly reduced upside but improved worst-window downside, making it useful as a risk-control mechanism. 

The highest-mean individual configuration was:

| Configuration | Mean Excess Return | Median | Worst Window | Best Window |
|---|---:|---:|---:|---:|
| Top-30, Huber, Rank Weighting, No Filter | **+1.643%** | +1.775% | -0.662% | +3.685% |
| Top-30, Huber, Rank Weighting, Filter | **+1.505%** | +1.695% | -0.260% | +2.889% |

For the Phase 1 competition submission, I selected the **filtered Top-30 Huber/rank configuration** because it preserved most of the expected upside while substantially reducing observed downside.

## Phase 2 Self-Ensemble

The Phase 1 live portfolio returned approximately **+6.99%**, outperforming the class average of roughly **+5.8%**, although it trailed the class-wide ensemble return of approximately **+7.8%**.

This suggested that the LightGBM pipeline contained useful signal but could benefit from additional variance reduction.

For Phase 2, I therefore constructed a **self-ensemble of several high-performing LightGBM portfolio configurations**.

The ensemble:

1. Generates several strong candidate portfolios.
2. Averages their portfolio weights.
3. Rewards stocks repeatedly selected by multiple models.
4. Retains the strongest consensus positions.
5. Caps individual weights at 10%.
6. Renormalizes the final portfolio.

The resulting ensemble used approximately 50 stocks at the portfolio level, providing more diversification than the aggressive Top-30 strategy while retaining exposure to high-consensus predictions. Also combats potential overfitting on previous stock data.

## Results

Historical performance was evaluated across four identical no-leakage windows for the original XGBoost baseline, the selected Phase 1 LightGBM model, and the Phase 2 self-ensemble.

| Model | Avg. Portfolio Return | Avg. Excess Return | Median Excess | Worst | Best | Hit Rate |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost Baseline | +3.316% | **+0.920%** | +0.922% | -1.275% | +3.109% | 75% |
| Phase 1 LightGBM | **+3.602%** | **+1.207%** | +1.174% | +0.192% | +2.287% | **100%** |
| Phase 2 Self-Ensemble | +3.435% | **+1.039%** | +1.032% | +0.122% | +1.970% | **100%** |

The Phase 1 model improved average excess return from **+0.920% to +1.207%** relative to the XGBoost baseline while eliminating negative excess-return windows in the historical test set.

The Phase 2 self-ensemble produced slightly lower mean excess return than the Phase 1 single model, but also achieved a **100% historical hit rate** while intentionally reducing reliance on any single aggressive configuration.

## Key Takeaways

This project evolved from a straightforward stock-return prediction task into a broader **portfolio-aware machine-learning system**.

The main lessons were:

- Benchmark-relative prediction benefited from **cross-sectional targets** rather than raw returns.
- Robust regression with **Huber loss** performed better than a direct ranking objective.
- Predictive performance alone was insufficient; the **portfolio construction rule materially affected realized results**.
- Concentrated Top-30 portfolios captured more alpha than highly diversified portfolios.
- Time-aware validation and embargo periods were critical for avoiding look-ahead leakage.
- Model and portfolio ensembling provided a practical way to reduce variance in a noisy financial prediction problem.

Across identical historical evaluation windows, the final LightGBM workflows consistently improved upon the provided XGBoost baseline while preserving a fully reproducible, no-leakage evaluation pipeline.


### File Directory
**fetch_extra_data.py** <br>
Supplemental-data helper. It can fetch valuation, financial, and macro data and merge available features into the panel. Financials are shifted with an announcement lag to reduce leakage risk.

**final_self_ensemble.csv** <br>
The final portfolio submission to the competition. csv file of 50 stocks with their respective weights

**generate_and_ensemble_candidates.py** <br>
Final Phase 2 ensemble generator. It trains/generates several candidate portfolios, validates each one, averages candidate weights, keeps top aggregate names, re-caps, normalizes, and validates the final ensemble.

**model_v2_portfolio_aware_safe.py** <br>
Final safe single-configuration model script. It adds portfolio-aware options such as top-k, filtering, weighting method, LambdaRank comparison, and label-safe historical as-of behavior.

**self_test_portfolios.py** <br>


**sweep_portfolio_aware_checked.py** <br>
Runs the 240-configuration sweep by repeatedly calling the safe model, validating portfolios, scoring with score_submission.py, and recording results. It checks ASOF < START and valid trading dates.
