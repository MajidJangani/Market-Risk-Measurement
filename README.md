# Market Risk Measurement Of Equity Portfolio 

## Live Project
View the complete project analysis and implementation at [GitHub Pages](https://majidjangani.github.io/https://majidjangani.github.io/Market-Risk-Measurement/)

## Objectives
- **Build a diversified equity portfolio**: Create a portfolio of stocks from various industries using historical price data.
- **Measure risk using VaR and CVaR** : Assess potential portfolio losses using both Value at Risk (VaR) and Conditional Value at Risk (CVaR) methods.
Optimize the portfolio: Use Monte Carlo simulations to identify the best risk-adjusted portfolio allocation.
- **Evaluate portfolio performance under stress**: Conduct stress testing and sensitivity analysis to understand the portfolio's response to market shocks and extreme conditions.
- **Ensure regulatory compliance** : Consider regulatory frameworks like Basel to ensure the portfolio adheres to risk management guidelines.
Methodology
-**Data Collection** : Gather historical stock price data from various companies across different industries using the yfinance library. The dataset covers June 1, 2023, to December 31, 2024.
- **Portfolio Construction** : Build a portfolio by selecting a range of stocks, and assign them random weights to form multiple portfolio combinations.
- **Risk Assessment** : Use both historical and Monte Carlo simulation methods to calculate VaR and CVaR, measuring the potential downside risk under normal and extreme market conditions.
- **Portfolio Optimization** : Run Monte Carlo simulations to identify the optimal risk-return trade-off by analyzing different portfolio weights and their respective performance metrics.
- **Stress Testing** : Simulate how the portfolio would behave under market shocks or extreme events to assess its resilience.
- **Sensitivity Analysis** : Evaluate how changes in individual stock prices or external factors impact the portfolio's overall risk profile.
## Libraries Used:
NumPy -Pandas - yFinance - Scipy - Seaborn - Matplotlib 

## Repository Structure
```python
MArket-Risk-Measurement-/
root/
/
├── _includes/
│   └── head-custom.html
├── assets/
│   └── style.scss
├── Market-Risk-Analytics.md 
├── README.md
├── _config.yml
└── index.md

