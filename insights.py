# insights.py
import pandas as pd
import os

def get_top_performers():
    metrics = pd.read_csv('data/risk_metrics.csv', index_col=0)
    
    print("🏆 TOP PERFORMERS")
    print("-" * 50)
    print(f"Best Return: {metrics['returns_annualized_return'].idxmax()} "
          f"({metrics['returns_annualized_return'].max():.1%})")
    print(f"Best Risk-Adjusted: {metrics['sharpe_ratio'].idxmax()} "
          f"(Sharpe: {metrics['sharpe_ratio'].max():.2f})")
    print(f"Lowest Risk: {metrics['volatility_annual_volatility'].idxmin()} "
          f"(Vol: {metrics['volatility_annual_volatility'].min():.1%})")

if __name__ == "__main__":
    get_top_performers()