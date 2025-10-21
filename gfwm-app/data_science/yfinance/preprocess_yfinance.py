import pandas as pd
import yfinance as yf
import numpy as np

data = pd.read_csv("SP 500 ESG Risk Ratings.csv")
data = data.dropna()

# Gets unique stock symbols from the data
unique_symbols = data['Symbol'].unique()

data["Growth Estimate"] = 0
data["Annual Return"] = 0
data["Volatility"] = 0

# constant used to annualize monthly volatility
sqrt_T = np.sqrt(12)

# Function to calculate growth estimate, annual return, and volatility for each ticker

# Function to calculate multiple new columns


def calculate_metrics(row):
    # Initialize variables
    growth_estimate_5yr = 0
    annual_return = 0
    volatility = 0

    stock = yf.Ticker(row["Symbol"])

    # Retrieves growth estimates data from yfinance
    growth_estimates = stock.growth_estimates

    # Checks if growth estimates data is available
    if growth_estimates is not None and '+5y' in growth_estimates.index:
        # Extracts the growth estimate for the next 5 years in the 'stock' column
        growth_estimate_5yr = growth_estimates.loc['+5y', 'stock']

    # next use 10y of prices to calculate volatility

    # https://github.com/ranaroussi/yfinance/wiki/Ticker
    monthly_prices = stock.history(interval="1mo", period="10y", actions=True)

    # https://www.macroption.com/historical-volatility-calculation/#log-returns
    monthly_prices["Log Return"] = np.log(
        monthly_prices['Close'] / monthly_prices['Close'].shift(1))

    # ddof = 1 uses Bessel's correction, ie / n-1
    volatility = sqrt_T * np.nanstd(monthly_prices['Log Return'], ddof=1)

    # https://www.investopedia.com/terms/a/annualized-total-return.asp
    cumulative_return = monthly_prices['Close'].iloc[-1] / monthly_prices['Close'].iloc[0]
    
    dates = monthly_prices.index
    days_held = (dates[-1] - dates[0]).days
    
    #print(days_held, volatility, cumulative_return)
    
    annual_return = pow(cumulative_return, 365/days_held) - 1
    
    print(growth_estimate_5yr, annual_return, volatility)

    # change to percent
    return pd.Series([growth_estimate_5yr, annual_return * 100, volatility * 100])

data[['Growth Estimate', 'Annual Return', 'Volatility']] = data.apply(calculate_metrics, axis=1)

# note: i don't think it makes sense to normalize the data we will be displaying

# Normalize the growth estimates to [0, 10]
# min = data['Growth Estimate'].min()
# max = data['Growth Estimate'].max()

# data['Growth Estimate'] = 10 * (data['Growth Estimate'] - min) / (max - min)

# Save the DataFrame to a CSV file
# Set index=False to avoid writing row indices
data.to_csv('preprocessed.csv', index=False)
