import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from data_science import stock_filter

# call filter stocks to come up with universe of 100 stocks
# tickers = stock_filter.filter_stocks(5, 5, 5)["ticker"]


# read csv for return and cov matrix
all_price_data = pd.read_csv("data_science/quant/sp500_daily_data.csv")[["ticker", "log_return", "volatility"]]

entire_cov_matrix = pd.read_csv("data_science/quant/sp500_covariance_matrix.csv")
entire_cov_matrix = entire_cov_matrix.set_index('ticker')

unique_tickers = pd.Series(all_price_data['ticker'].unique())

# https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/

# Function to generate a plot
def generate_plot(ax, i):
    num_tickers = 3 + (i//3)
    tickers = unique_tickers.sample(frac=100/500, random_state=i).reset_index(drop = True)
    
    # keep selected tickers
    price_data = all_price_data[all_price_data['ticker'].isin(tickers)]

    # keep entries where both tickers are present
    cov_matrix = entire_cov_matrix.loc[tickers, tickers].to_numpy()


    mean_log_returns = price_data.groupby(by="ticker").log_return.agg("mean")
    
    # Returns the mean and standard deviation of returns for a random portfolio
    def random_portfolio():
        rand_weights = np.random.rand(len(tickers))
        rand_weights = rand_weights / sum(rand_weights)

        est_return = np.dot(rand_weights.T, mean_log_returns) * 252

        sigma = np.sqrt(
            np.matmul(rand_weights.T, np.matmul(cov_matrix, rand_weights)))

        # This recursion reduces outliers to keep plots pretty
        if sigma > 0.03:
            return random_portfolio()
        return est_return, sigma

    n_portfolios = 2000
    means, stds = np.column_stack([
        random_portfolio()
        for _ in range(n_portfolios)
    ])
    
    ax.scatter(stds, means, s = 2, alpha = 0.7)
    
    #formatted_tickers = ', '.join(tickers)
    #ax.set_title(f'Tickers: {formatted_tickers}')
    ax.set_title("Portfolio of 100 random SP 500 component stocks")

# Define the grid size
nrows = 2
ncols = 2

# Create subplots
fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10))

# Populate the subplots using a for loop
for i in range(0, nrows*ncols):
    row = i // ncols
    col = i % ncols
    generate_plot(axs[row, col], i)

plt.tight_layout()
plt.show()

# py.iplot_mpl(fig, filename='mean_std', strip_style=True)
