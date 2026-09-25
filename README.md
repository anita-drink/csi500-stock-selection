# csi500-stock-selection DESC IS WIP
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
