import pandas as pd
import numpy as np

from data_science.quant.portfolio_calculator import ANNUAL_RISK_FREE_RATE

def extract_value(var):
    if isinstance(var, list) and len(var) == 1:
        return var[0]
    return var

# TODO handle include_spy = False better
def portfolio_history(portfolio, include_spy = True):
    
    spy_log_returns = pd.read_csv("data_science/quant/spy_timeseries_13-24.csv")['SPY']
    
    tickers = portfolio.index.tolist()
    
    tickers_log_returns = pd.read_csv("data_science/quant/sp500_timeseries_13-24.csv")[['date'] + tickers]
    # print(tickers_log_returns.head())
    
    # TODO clean this up using pandas objects instead of python lists
    
    # print(len(spy_log_returns))
    # print(len(tickers_log_returns))
    days = len(spy_log_returns)
    # major assumption: since length is the same, the days automatically line up
    
    init_value = 100
    cash_daily_return = np.log(1+ANNUAL_RISK_FREE_RATE) / 252
    
    # cash position earning risk free rate is the same for spy and portfolio:
    t = np.arange(start = 0, stop = days + 1)
    cash_percent =  1- (portfolio['weight'].sum())
    
    cash_portion = (init_value * cash_percent) * np.exp((cash_daily_return) * t)  # Exponential growth formula
    
    # calculate spy growth
    
    spy_timeseries = [init_value * (1-cash_percent)]
     
    for log_return in spy_log_returns.values:
        spy_timeseries.append(spy_timeseries[-1] * np.exp(log_return))
        
        
    ticker_timeseries = {}
    
    for ticker in tickers:
        
        timeseries = [init_value]
        
        for log_return in tickers_log_returns[ticker].values:
            log_return = extract_value(log_return)
            # print(log_return)
            if pd.isna(log_return):
                log_return = cash_daily_return
            timeseries.append(timeseries[-1] * np.exp(log_return))
        
        # get weight for current ticker
        weight = portfolio.loc[ticker]['weight']
        
        # sometimes weight is a list... TODO figure out why
        if isinstance(weight, (list, np.ndarray, pd.Series)) and len(weight) == 1:
            weight = weight[0]
            
        ticker_timeseries[ticker] = [value * weight for value in timeseries]
    
    # initially all 0's. 
    # days + 1 because start is init_value, then data actually starts
    portfolio_timeseries = [0] * (days + 1)
    # add portfolios one by one, elementwise, to result
    for timeseries in ticker_timeseries.values():
        portfolio_timeseries = [p + t for p, t in zip(portfolio_timeseries, timeseries)]
    
    
    # add cash to both portfolios
    spy_timeseries = [spy + cash for spy, cash in zip(spy_timeseries, cash_portion)]
    portfolio_timeseries = [p + cash for p, cash in zip(portfolio_timeseries, cash_portion)]
        
    # calculate max drawdown as a percent
    # https://quant.stackexchange.com/a/43544/78596
    def get_max_drawdown(nvs: pd.Series, window=None) -> float:
        """
        :param nvs: net value series
        :param window: lookback window, int or None
        if None, look back entire history
        """
        n = len(nvs)
        if window is None:
            window = n
        # rolling peak values
        peak_series = nvs.rolling(window=window, min_periods=1).max()
        return (nvs / peak_series - 1.0).min()
    
    spy_max_drawdown = get_max_drawdown(pd.Series(spy_timeseries))
    portfolio_max_drawdown = get_max_drawdown(pd.Series(portfolio_timeseries))
    
    if(include_spy):
        return spy_timeseries[::5], portfolio_timeseries[::5], tickers_log_returns['date'][::5].tolist(), spy_max_drawdown, portfolio_max_drawdown
    else:
        return portfolio_timeseries[::5], tickers_log_returns['date'][::5].tolist(), portfolio_max_drawdown
        


# d = {'ticker': ['ABNB', 'AAPL', 'MSFT'], 'weight': [0.3, 0.3, 0.4]}
# df = pd.DataFrame(data=d)
# df.set_index('ticker', drop = True, inplace=True)

# portfolio_history(df)