import pandas as pd
import numpy as np
from math import floor

from functools import reduce


def filter_stocks(user_preferences, count=100, flexibility=0, tickers_only=False):
    
    """
    Filters the ESG dataset based off the user_preferences dict.
    
    Args:
        user_preferences (dict): Must include the following keys: "environment", 
                                    "human_rights", "community", "workforce", "product_responsibility", 
                                    "shareholders", "management",
                                    'avoid_fossil_fuels', 'avoid_weapons'
        count (int): How many stocks to return
        flexibility (int 0 - 100): Percent to extend the acceptable threshold
        tickers_only (boolean): Whether to return a series of only tickers or dataframe with more columns
        
    Returns:
        (df['ticker', 'compatibility', 'environment, 'social, 'governance']):\n
        One row per ticker in the S&P 500, sorted by compatibility score
    """
    
    # already preprocessed to contain only stocks for which we have financial data
    # contains averages in various esg columns as well as fossil fuels & weapons yes/no data
    data = pd.read_csv("data_science/preprocess_and_filter/preprocessed_refinitiv.csv")
    
    #print(data)    
    
    # masks to use when excluding fossil fuels and/or weapons
    exclusion_masks = []
    
    compatibility_penalties = {}
    
    rated_factors = []
    
    # split the user preferences by primary and secondary
    for factor, value in user_preferences.items():
        
        if value == 10:
            rated_factors.append(factor)
            # lose points for lower ranking in important factors
            compatibility_penalties[factor] = 10
        
        elif value == 5:
            rated_factors.append(factor)
            # lose less points for lower ranking in mid importance factors
            compatibility_penalties[factor] = 5
        
        elif value == True:
            #means exclude this factor (either fossil fuels or weapons)
            column_name = factor.split('avoid_')[1]
            exclusion_masks.append(data[column_name].to_numpy())
        else:
            # don't lose points in factors ranked not important
            compatibility_penalties[factor] = 0
    combined_exclusion_mask = np.ones(len(data), dtype = bool)
    #combine exclusion masks, not the result to be used as a filter
    if(len(exclusion_masks) > 0):
        # sum(exclusion_masks) adds the masks elementwise. 0s mean no issues, keep!
        combined_exclusion_mask = np.array([s==0 for s in sum(exclusion_masks)])
    
    #calculate compatibility scores for all stocks
    # calculating it after filtering would rank within the filtered results
    # use decile ranks within important columns
    
    def calculate_row_compatibility(row):
        # rank each factor within its column, in descending order
        ranks = {factor: data[factor].rank(ascending=False, pct=True)[row.name] for factor in rated_factors}
    
        # comptibility calculation:
        # calculate how far the company is down the ranking, taking into account flexibility
        # with flexibility 0, rank  <= 15% results in 0 penalty
        #                      15 < rank <= 25 results in 1 penalty
        #                         etc each 10% increase in rank increases penalty by 1
        # ex: ABC ranks in the 42% percentile with flexibility 15
        # taking into account flexibility, the highest rank to still hold no penalty is 25%
        # ABC is 2 levels up from that: 25-35, 35-45
        # https://www.desmos.com/calculator/lfyjozqfcn
        
        unweighted_penalties = {factor: max(0, floor(10*(r - flexibility) - 0.501)) for factor, r in ranks.items()}
        
        # then each of these penalties is weighed by that factor's importance to calculate the total penalty
        
        score = 100
        for factor in rated_factors:
            score = score - compatibility_penalties[factor] * unweighted_penalties[factor]
        return max(0, score)
        
    # apply comptability calculation described to each row
    data['compatibility'] = data.apply(calculate_row_compatibility, axis = 1)
    
    # now filter out rows that client wants to exclude
    data = data[combined_exclusion_mask]
    
    # sort by compatibility score
    data.sort_values(by='compatibility', ascending=False, inplace=True)
    
    results = data[['ticker', 'compatibility', 'environment', 'social', 'governance']]
    
    if (tickers_only):
        return results['ticker'].head(count)

    return results

# only returns tickers
def filter_stocks_mass(user_preference_dicts, count=100, flexibility=0):

    data = pd.read_csv("../preprocessed_refinitiv.csv")

    quantile_threshold = 0.5 * (1 - flexibility/100.0)
    
    factors = ["environment", "human_rights", "workforce", 
               "product_responsibility", "shareholders", "community", "management"]
    quantiles = {factor: data[factor].quantile(quantile_threshold) for factor in factors}
    
    print(quantiles)
    
    result = []
    
    for user_preference_dict in user_preference_dicts:
        
        #select the factors that have a value of 10
        top_factors = {factor:quantiles[factor] for factor, value in user_preference_dict.items() if value == 10}

        # filter rows where the selected factors are higher than their quantiles
        masks = [data[factor] >= quantile for factor, quantile in top_factors.items()]
        # combine masks using element-wise AND
        combined_mask = reduce(lambda mask1, mask2: [
                            el1 and el2 for el1, el2 in zip(mask1, mask2)], masks)
        selected_tickers = data[combined_mask].reset_index(drop=True)['ticker']
        
        combination = ', '.join(top_factors.keys())
        result.append((combination, selected_tickers))

    return result

#tests
#results: 109 primary results, 
# secondary has 16 for product_responsibility, 31 for management
'''test_dict = {'environment':10, 
             'human_rights': 10,  
             'workforce':1,
             'product_responsibility':5,
             'shareholders':1,
             'community':10,
             'management':5}
primary, secondary = filter_stocks(test_dict)

print(primary.shape)
#print(primary[['ticker', 'environment', 'human_rights', 'community', 'compatibility']])
for factor, df in secondary.items():
    print(factor, df.shape)'''
