"""
Portfolio Optimization Module
Finds optimal portfolio weights using Modern Portfolio Theory
"""

import pandas as pd
import numpy as np
import logging
from scipy.optimize import minimize
from config import PORTFOLIO_PARAMS, RISK_FREE_RATE

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Class for portfolio optimization"""
    
    def __init__(self, returns_df, risk_free_rate=RISK_FREE_RATE):
        """
        Initialize portfolio optimizer
        
        Args:
            returns_df: DataFrame of daily returns for multiple stocks
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns_df
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns_df.columns)
        self.assets = returns_df.columns.tolist()
        
        # Calculate annualized metrics
        self.mean_returns = returns_df.mean() * 252
        self.cov_matrix = returns_df.cov() * 252
        
    def portfolio_performance(self, weights):
        """
        Calculate portfolio return and volatility
        
        Args:
            weights: Portfolio weights array
            
        Returns:
            Tuple of (return, volatility, Sharpe ratio)
        """
        returns = np.sum(self.mean_returns * weights)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (returns - self.risk_free_rate) / volatility if volatility > 0 else 0
        return returns, volatility, sharpe
    
    def negative_sharpe(self, weights):
        """Negative Sharpe ratio (for minimization)"""
        return -self.portfolio_performance(weights)[2]
    
    def check_sum(self, weights):
        """Constraint: sum of weights = 1"""
        return np.sum(weights) - 1
    
    def maximize_sharpe_ratio(self, min_weight=0.05, max_weight=0.40):
        """
        Find portfolio with maximum Sharpe ratio
        
        Args:
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            
        Returns:
            Dictionary with optimal weights and performance metrics
        """
        # Initial guess (equal weights)
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': self.check_sum})
        
        # Optimize
        result = minimize(
            self.negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            ret, vol, sharpe = self.portfolio_performance(optimal_weights)
            
            return {
                'weights': dict(zip(self.assets, optimal_weights)),
                'expected_return': ret,
                'expected_volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            logger.error("Optimization failed: " + result.message)
            return {'success': False, 'message': result.message}
    
    def minimize_volatility(self):
        """
        Find minimum variance portfolio
        
        Returns:
            Dictionary with optimal weights and performance
        """
        # Initial guess
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': self.check_sum})
        
        # Objective: minimize volatility
        def portfolio_volatility(weights):
            return self.portfolio_performance(weights)[1]
        
        result = minimize(
            portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            ret, vol, sharpe = self.portfolio_performance(optimal_weights)
            
            return {
                'weights': dict(zip(self.assets, optimal_weights)),
                'expected_return': ret,
                'expected_volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False}
    
    def efficient_frontier(self, points=50):
        """
        Calculate efficient frontier points
        
        Args:
            points: Number of points on the frontier
            
        Returns:
            DataFrame with returns and volatilities
        """
        target_returns = np.linspace(
            self.mean_returns.min(),
            self.mean_returns.max(),
            points
        )
        
        efficient_portfolios = []
        
        for target in target_returns:
            constraints = (
                {'type': 'eq', 'fun': self.check_sum},
                {'type': 'eq', 'fun': lambda w: np.sum(self.mean_returns * w) - target}
            )
            
            result = minimize(
                lambda w: self.portfolio_performance(w)[1],
                np.array([1/self.n_assets] * self.n_assets),
                method='SLSQP',
                bounds=tuple((0, 1) for _ in range(self.n_assets)),
                constraints=constraints
            )
            
            if result.success:
                ret, vol, _ = self.portfolio_performance(result.x)
                efficient_portfolios.append({
                    'return': ret,
                    'volatility': vol
                })
        
        return pd.DataFrame(efficient_portfolios)
    
    def generate_random_portfolios(self, n_portfolios=1000):
        """
        Generate random portfolios for visualization
        
        Args:
            n_portfolios: Number of random portfolios
            
        Returns:
            DataFrame with portfolio metrics
        """
        results = []
        
        for _ in range(n_portfolios):
            weights = np.random.random(self.n_assets)
            weights = weights / np.sum(weights)
            
            ret, vol, sharpe = self.portfolio_performance(weights)
            
            results.append({
                'return': ret,
                'volatility': vol,
                'sharpe': sharpe,
                'weights': weights
            })
        
        return pd.DataFrame(results)
    
    def get_portfolio_summary(self, weights_dict):
        """
        Get detailed summary for a specific portfolio
        
        Args:
            weights_dict: Dictionary of {symbol: weight}
            
        Returns:
            DataFrame with contribution analysis
        """
        weights = np.array([weights_dict[asset] for asset in self.assets])
        ret, vol, sharpe = self.portfolio_performance(weights)
        
        # Calculate individual contributions
        contributions = []
        for i, asset in enumerate(self.assets):
            weight = weights[i]
            asset_return = self.mean_returns.iloc[i]
            
            # Marginal contribution to risk
            marginal_risk = (self.cov_matrix.iloc[i].dot(weights)) / vol
            contribution_to_risk = weight * marginal_risk
            
            contributions.append({
                'Asset': asset,
                'Weight': f'{weight:.2%}',
                'Return': f'{asset_return:.2%}',
                'Risk Contribution': f'{contribution_to_risk:.2%}',
                'Risk %': f'{(contribution_to_risk/vol):.2%}'
            })
        
        summary = pd.DataFrame(contributions)
        
        # Add portfolio totals
        summary.loc['PORTFOLIO'] = [
            'TOTAL',
            f'{np.sum(weights):.2%}',
            f'{ret:.2%}',
            f'{vol:.2%}',
            '100%'
        ]
        
        return summary


def optimize_portfolio(stock_data_dict, method='max_sharpe'):
    """
    Convenience function for portfolio optimization
    
    Args:
        stock_data_dict: Dictionary of stock DataFrames
        method: 'max_sharpe' or 'min_volatility'
        
    Returns:
        Optimization results
    """
    # Calculate returns
    returns_df = pd.DataFrame()
    for symbol, df in stock_data_dict.items():
        returns_df[symbol] = df['close'].pct_change()
    
    returns_df = returns_df.dropna()
    
    optimizer = PortfolioOptimizer(returns_df)
    
    if method == 'max_sharpe':
        return optimizer.maximize_sharpe_ratio()
    elif method == 'min_volatility':
        return optimizer.minimize_volatility()
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # Test portfolio optimization
    print("Testing Portfolio Optimization...")
    
    # Create sample returns data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    returns_df = pd.DataFrame({
        'AAPL': np.random.randn(len(dates)) * 0.02,
        'JPM': np.random.randn(len(dates)) * 0.025,
        'JNJ': np.random.randn(len(dates)) * 0.015,
        'MSFT': np.random.randn(len(dates)) * 0.018,
    }, index=dates)
    
    optimizer = PortfolioOptimizer(returns_df)
    
    print("\n1. Max Sharpe Portfolio:")
    result = optimizer.maximize_sharpe_ratio()
    if result['success']:
        for asset, weight in result['weights'].items():
            print(f"   {asset}: {weight:.2%}")
        print(f"   Expected Return: {result['expected_return']:.2%}")
        print(f"   Expected Volatility: {result['expected_volatility']:.2%}")
        print(f"   Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    
    print("\n2. Random Portfolios:")
    random_portfolios = optimizer.generate_random_portfolios(10)
    print(random_portfolios[['return', 'volatility', 'sharpe']].head())