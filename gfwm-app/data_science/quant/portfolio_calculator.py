import pandas as pd
import numpy as np
import data_science.quant.baseline_markowitz.markowitz_optimization as markowitz_optimization
import data_science.quant.optimized_markowitz.bootstrapped_markowitz as opt_markowitz

# TODO move this to .env
# risk free rate based of historical average of 30d yield
ANNUAL_RISK_FREE_RATE = 0.0153

def map_risk_appetite_to_cash_percent(risk_appetite):
    # risk_appetite ∈ [0, 20]
    # 0 → 80% cash, 20 → 0% cash
    return 0.80 - 4 * risk_appetite


def calculate_portfolio(ticker_compatibility_df, cash_percent, use_baseline_markowitz, use_optimized_markowitz, return_summary_statistics = True):

    performance_summaries = pd.read_csv("data_science/data/universal_data/sp500_performance_summaries.csv")
    tickers = ticker_compatibility_df['ticker']
    n = len(tickers)
    annual_returns = performance_summaries[tickers].iloc[0]

    if (use_baseline_markowitz):
        # Use the baselline Markowitz from 2024's Group

        # TODO move this check to filter stocks so it does not return tickers for which we don't have financial data
        # important note: tickers are in the same order in every data source

        adjusted_cov_matrix = pd.read_csv(
            "data_science/data/covariance_models/baseline_markowitz/sp500_adjusted_cov_matrix.csv")
        adjusted_cov_matrix.set_index('ticker', inplace=True)

        true_cov_matrix = pd.read_csv(
            "data_science/data/covariance_models/baseline_markowitz/sp500_raw_cov_matrix.csv")
        true_cov_matrix.set_index('ticker', inplace=True)

        # keep entries where both tickers are present
        adjusted_cov_matrix = adjusted_cov_matrix.loc[tickers, tickers].to_numpy()
        true_cov_matrix = true_cov_matrix.loc[tickers, tickers].to_numpy()

        # keep identified tickers and convert to daily mean log returns
        mean_log_returns = annual_returns/252
        # print(mean_log_returns * 252)

        # daily target returns starting from risk free rate to 20% annually
        # scale to annual for graphing

        # efficiency improvement:
        # calculate fewer target returns in the range most likely to contain the tangency portfolio
        # the tangent portfolio has returns around 15%, only worth calculating target returns in that range
        target_returns = np.linspace(start=0.05/252,
                                     stop=0.2/252, num=25)

        bounds = [0.5/n, 3.0/n]
        #TODO speed up: precalculate the markowitz ideal portfolio for all the combinations of factors
        # then run markowitz again after the client edits their portfolio and chooses to recalculate the weights
        markowitz_portfolios = markowitz_optimization.calculate_optimal_portfolios(true_mean_returns=mean_log_returns,
                                                                                   adjusted_mean_returns=mean_log_returns,
                                                                                   true_cov=true_cov_matrix,
                                                                                   adjusted_cov=adjusted_cov_matrix,
                                                                                   target_returns=target_returns,
                                                                                   annual_risk_free_rate=ANNUAL_RISK_FREE_RATE,
                                                                                   bounds=bounds)

        # idea: calculate the slope of the line between (0, rfr), (point)
        # the steepest slope is the ideal portfolio we want

        # slope = (return - rfr) / volatility
        # same as picking the portfolio with the largest sharpe!

        markowitz_portfolios['slope'] = (
            markowitz_portfolios['annual_return'] - ANNUAL_RISK_FREE_RATE) / markowitz_portfolios['annual_volatility']

        tangent_portfolio_index = markowitz_portfolios['slope'].argmax()

        tangent_portfolio = markowitz_portfolios.iloc[tangent_portfolio_index]

        # interpolate the line between (0, risk_free_rate) and the tangent portfolio
        
        # cash percent is 1-alpha

        alpha = 1 - cash_percent
        
        ideal_weights = np.array(alpha * tangent_portfolio['weights']).reshape(n, 1)
    
    elif (use_optimized_markowitz):
        adj_cov_df = pd.read_csv(
            "data_science/data/covariance_models/optimized_markowitz/sp500_adjusted_cov_matrix.csv",
            index_col=0,
        )
        true_cov_df = pd.read_csv(
            "data_science/data/covariance_models/optimized_markowitz/sp500_raw_cov_matrix.csv",
            index_col=0,
        )

        # --- Align tickers: drop any tickers that aren't in the optimized Σ ---
        available = [t for t in tickers if t in adj_cov_df.index]

        if len(available) == 0:
            raise ValueError("None of the requested tickers are present in the optimized covariance matrices.")

        if len(available) != len(tickers):
            missing = [t for t in tickers if t not in adj_cov_df.index]
            print(f"[WARN] Dropping {len(missing)} tickers not in optimized Σ: {missing}")

        # overwrite tickers / annual_returns / n to match the optimized universe
        tickers = pd.Index(available)
        annual_returns = annual_returns[tickers]
        n = len(tickers)

        # keep only the tickers we’re actually using, in the same order
        adjusted_cov_matrix = adj_cov_df.loc[tickers, tickers].to_numpy()
        true_cov_matrix = true_cov_df.loc[tickers, tickers].to_numpy()

        # --- Load daily returns for these tickers ---
        returns_df_all = pd.read_csv(
            "data_science/data/universal_data/sp500_timeseries_13-24.csv",
            index_col=0,
            parse_dates=True,
        )
        returns_df = returns_df_all[tickers]   # same column order as tickers

        lb_vec, ub_vec = opt_markowitz.make_bounds_from_sigma(
        adjusted_cov_matrix,
        max_cap=0.20,   # was 0.05   <-- CHANGED
        k=10.0,         # was 5.0    <-- CHANGED (more weight allowed to low-vol names)
        floor_frac=0.5, # same
        )

    # ensure min-variance portfolio is feasible (lift caps if needed, cap hard at 30% instead of 10%)
        w_minvar = opt_markowitz._qp_min_variance(
        adjusted_cov_matrix,
        lb_vec,
        np.ones_like(ub_vec),
        )
        ub_vec = opt_markowitz.lift_caps_above_minvar(
        ub_vec,
        w_minvar,
        slack=1.0,      # was 1.2    <-- CHANGED (don’t tighten further)
        hard_cap=0.30,  # was 0.10   <-- CHANGED
        )

    # extra safety: make sure total capacity can actually reach 1
        if ub_vec.sum() < 1.0:
            scale = 1.05 * (1.0 / ub_vec.sum())
            ub_vec *= scale
            print(f"[WARN] sum(ub_vec) was < 1, scaled bounds up by factor {scale:.3f}")

        # --- Build target return grid (daily), constrained by expected ranges ---
        mu_daily = (annual_returns / 252.0).values  # shape (n,)
        rf_daily = (1.0 + ANNUAL_RISK_FREE_RATE) ** (1.0 / 252.0) - 1.0

        # Try to get a feasible range via QP; if it blows up, fall back.
        try:
            rmin, rmax = opt_markowitz.feasible_return_range(mu_daily, lb_vec, ub_vec)
        except Exception as e:
            print(f"[WARN] feasible_return_range failed: {e}. Falling back to percentile-based target grid.")
            rmin, rmax = None, None

        if (rmin is not None) and (rmax is not None) and (rmax > rf_daily):
            lo = max(rf_daily, rmin + 1e-8)
            hi = max(lo + 1e-8, rmax - 1e-8)
            target_returns = np.linspace(lo, hi, 60)
        else:
            # fallback: percentile grid on annual μ, same idea as optimized script
            mu_annual = annual_returns.values
            mu_clean = mu_annual[~np.isnan(mu_annual)]
            min_ann = float(np.nanpercentile(mu_clean, 20))
            max_ann = float(np.nanpercentile(mu_clean, 80) * 1.5)
            target_returns = np.linspace(min_ann / 252.0, max_ann / 252.0, 60)
        # --- Run Michaud bootstrap resampling ---
        base_frontier, michaud_frontier = opt_markowitz.michaud_resampled_frontier(
            returns_df=returns_df,
            true_cov=true_cov_matrix,
            adjusted_cov=adjusted_cov_matrix,
            target_returns=target_returns,
            annual_risk_free_rate=ANNUAL_RISK_FREE_RATE,
            bounds=(lb_vec, ub_vec),
            n_resamples=80,
            seed=123,
        )

        # --- Tangent (max-sharpe) portfolio ---
        tangent = opt_markowitz.compute_tangent_from_frontier(
            frontier_df=michaud_frontier,
            annual_risk_free_rate=ANNUAL_RISK_FREE_RATE,
        )

        # --- Scale weights by alpha = 1 - cash_percent (client’s risk appetite) ---
        alpha = 1.0 - cash_percent
        w_tangent = np.asarray(tangent["weights"]).ravel()
        ideal_weights = (alpha * w_tangent).reshape(n, 1)
    else:
        # <<< FIXED: previously true_cov_matrix was undefined here

        alpha = 1 - cash_percent
        ideal_weights = np.repeat(alpha / n, n).reshape(n, 1)

        # Load any covariance matrix so summary statistics will NOT crash
        true_cov_df = pd.read_csv(
            "data_science/data/covariance_models/baseline_markowitz/sp500_raw_cov_matrix.csv"
        ).set_index("ticker")

        true_cov_matrix = true_cov_df.loc[tickers, tickers].to_numpy()  # <<< NEW
    
    if(not return_summary_statistics):
        # convert to row vector of weights
        return ideal_weights.T.tolist()[0]
    
    expected_return = np.dot(ideal_weights.reshape(n,), annual_returns) + cash_percent * ANNUAL_RISK_FREE_RATE

    expected_volatility = np.sqrt(252 * 
                                  (ideal_weights.T @ true_cov_matrix @ ideal_weights)[0][0])

    sharpe = (expected_return - ANNUAL_RISK_FREE_RATE) / expected_volatility

    return ideal_weights.T.tolist()[0], expected_return, expected_volatility, sharpe


def extract_value(var):
    if isinstance(var, list) and len(var) == 1:
        return var[0]
    return var

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

def portfolio_history(portfolio, include_spy = True):
    
    spy_log_returns = pd.read_csv("data_science/data/universal_data/spy_timeseries_13-24.csv")['SPY']
    
    if(portfolio.empty):
        tickers = []
    else:
        tickers = portfolio.index.tolist()
    
    print("tickers are ", tickers)
    
    #print("tickers are ", tickers)
    
    tickers_log_returns = pd.read_csv("data_science/data/universal_data/sp500_timeseries_13-24.csv")[['date'] + tickers]
    
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
    
    spy_max_drawdown = get_max_drawdown(pd.Series(spy_timeseries))
    portfolio_max_drawdown = get_max_drawdown(pd.Series(portfolio_timeseries))
    
    if(include_spy):
        return spy_timeseries, portfolio_timeseries, tickers_log_returns['date'].tolist(), spy_max_drawdown, portfolio_max_drawdown
    else:
        return portfolio_timeseries, tickers_log_returns['date'].tolist(), portfolio_max_drawdown
    
# d = {'ticker': ['ABNB', 'AAPL', 'MSFT'], 'weight': [0.3, 0.3, 0.4]}
# df = pd.DataFrame(data=d)
# df.set_index('ticker', drop = True, inplace=True)

# portfolio_history(df)

def calculate_summary_statistics(timeseries, return_as_range = False):
    '''
    Given a timeseries (Python list), calculate return range,  volatility, sharpe
    '''
    
    log_returns = np.log(np.array(timeseries[1:]) / np.array(timeseries[:-1]))
    
    mean_log_return = np.mean(log_returns)
    
    average_return = (np.exp(mean_log_return) - 1) * 252
    volatility = np.std(log_returns) * np.sqrt(252)
    
    sharpe = (average_return - ANNUAL_RISK_FREE_RATE) / volatility
    
    if return_as_range:

        # Margin of error uses z-score, which is 1.96 for 95% confidence interval
        # use sample standard deviation
        # moe = z-score * (std_dev / sqrt(n))
        margin_of_error = 1.96 * (np.std(log_returns, ddof=1) / np.sqrt(len(log_returns)))

        # Confidence interval for log return is mean +- moe
        # convert to percent annual return
        lower_bound = (np.exp(mean_log_return - margin_of_error) - 1) * 252 
        upper_bound = (np.exp(mean_log_return + margin_of_error) - 1) * 252
        
        return (lower_bound, upper_bound), volatility, sharpe
    
    return average_return, volatility, sharpe

def calculate_esg_score(portfolio):
    
    if(portfolio.empty):
        return 0
    
    data = pd.read_csv("data_science/preprocess_and_filter/preprocessed_refinitiv.csv")[['ticker', 'environment', 'social', 'governance']].set_index('ticker', drop = True)
    
    merged_data = data.merge(portfolio, how='inner', left_index=True, right_index=True)
    
    # Calculate the weighted esg score per row
    merged_data['weighted_score'] = (merged_data[['environment', 'social', 'governance']].sum(axis=1) / 3) * merged_data['weight']

    merged_data['weight'] / merged_data['weight'].sum()
    
    # Sum the weighted scores over all rows
    total_esg_score = merged_data['weighted_score'].sum()

    #print("Total Weighted Score:", total_esg_score)
    
    return total_esg_score