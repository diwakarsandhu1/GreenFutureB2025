import pandas as pd

data = pd.read_csv("data_science/data/raw_data/Refinitiv ESG Final Data for Analysis.csv")

tickers_to_keep = pd.read_csv("data_science/data/universal_data/tickers_to_keep.csv")['ticker']

performance_data = pd.read_csv("data_science/data/universal_data/sp500_performance_summaries.csv")

company_data = pd.read_csv("data_science/data/raw_data/company_data.csv")
company_data.set_index('ticker', inplace=True)

columns_to_keep = ['Symbol', 'Name', 'ESG Combined Score', 'ESG Controversies Score', 
                   'Environment Pillar Score', 'Social Pillar Score','Governance Pillar Score',
                   'Human Rights Score', 'Community Score', 'Workforce Score', 'Product Responsibility Score', 
                   'Shareholders Score','Management Score']

data = data[columns_to_keep]
# pandas prefers single word columns. lowercase for simplicity with variable names
data = data.rename(columns={"Symbol": "ticker",
                            "Name": "name",
                            "ESG Combined Score": "esg_combined",
                            "ESG Controversies Score": "controversy",
                            "Environment Pillar Score": "environment",
                            "Social Pillar Score": "social",
                            "Governance Pillar Score": "governance",
                            "Human Rights Score": "human_rights",
                            "Community Score": "community",
                            "Workforce Score": "workforce",
                            "Product Responsibility Score": "product_responsibility",
                            "Shareholders Score": "shareholders",
                            "Management Score": "management"
                            })

numeric_columns_to_average = data.columns[2:]

# Delete rows containing the value 'Unknown'
data = data[~data.eq('Unknown').any(axis=1)]

# only keep the tickers for which we have financial data (are in the sp500 as of 11/2024)
data = data[data['ticker'].isin(tickers_to_keep)]

# for each ticker, calculate summary scores in each of the relevant columns
# use exponential weighted average over the years for which we have data

def map_company_data(boolean_input):
    if boolean_input:
        return 1
    elif not boolean_input:
        return 0
    return -1

def ewma_summaries(group):
    # Set span based on group size
    span = min(len(group), 10)
    #TODO selecting columns inside here, after the group by, is deprecated
    last_row = group[numeric_columns_to_average].ewm(span = span).mean().iloc[-1].round(4)
    
    ticker = group['ticker'].iloc[0]
    # get additional data based off ticker
    additional_data = company_data.loc[ticker]
    
    last_row['name'] = additional_data['name']
    last_row['annual_return'] = performance_data[ticker].iloc[0]
    last_row['volatility'] = performance_data[ticker].iloc[1]
    last_row['fossil_fuels'] = map_company_data(additional_data['fossil_fuels'])
    last_row['weapons'] = map_company_data(additional_data['weapons'])
    last_row['tobacco'] = map_company_data(additional_data['tobacco'])
    
    return last_row

result = (
    data.groupby('ticker')
      .apply(ewma_summaries)
)

#print(result)

result.to_csv('data_science/preprocess_and_filter/preprocessed_refinitiv.csv')