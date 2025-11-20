import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from data_science.preprocess_and_filter.stock_filter import filter_stocks
import data_science.quant.portfolio_calculator as pc
import data_science.quant.monte_carlo.monte_carlo as mc 

@csrf_exempt
def hello_api(request):
    return JsonResponse({"message": f"Hello from the Django API! {request}"})

@csrf_exempt
def submit_form(request):
    if request.method == "POST":
        
        # build dict of client responses
        client_responses = json.loads(request.body)
        
        # check that required factors are present
        required_factors = ["environment", "human_rights", "community", "workforce",
                            "product_responsibility", "shareholders", "management"]
        
        avoid_factors = ['avoid_fossil_fuels', 'avoid_weapons']
        
        missing_keys = [key for key in required_factors + avoid_factors + ['risk_appetite']
                        if key not in client_responses.keys()]
        if len(missing_keys) > 0:
            return JsonResponse({"error": f"Missing keys: {missing_keys}"}, status=400)
        
        # prepare client responses for filtering and markowitz
        
        # drag and drop factors
        esg_preferences = {key: value for (key, value) in client_responses.items() if key in required_factors}
        
        # avoid factors (checkbox)
        for avoid_factor in avoid_factors:
            esg_preferences[avoid_factor] = client_responses[avoid_factor]
        # map risk appetite (0 - 0.2) to cash percent (50% - 10%)
        cash_percent = pc.map_risk_appetite_to_cash_percent(client_responses['risk_appetite'])
        
        use_baseline_markowitz = (client_responses['weighing_scheme'] == 'Baseline Markowitz')
        use_optimized_markowitz = (client_responses['weighing_scheme'] == 'Optimized Markowitz')
        
        # filter stocks using client responses
        # df with two columns: ticker, compatibility
        filter_results = filter_stocks(user_preferences= esg_preferences)
        
        # this value determines how many stocks to include in portfolio
        portfolio = filter_results.head(100).copy()
        
        # calculate best fit portfolio for the client
        ideal_portfolio_weights = pc.calculate_portfolio(portfolio[['ticker', 'compatibility']],
                                                      cash_percent,
                                                      use_baseline_markowitz,
                                                      use_optimized_markowitz,
                                                      return_summary_statistics=False)
        
        portfolio['weight'] = ideal_portfolio_weights
        
        spy_timeseries, portfolio_timeseries, dates, spy_max_dd, portfolio_max_dd = pc.portfolio_history(portfolio[['ticker',
                                                                                                                 'weight']]
                                                                                                      .set_index('ticker', 
                                                                                                                 drop = True))
        
        #calculate summary statistics  
        
        portfolio_return, portfolio_volatility, portfolio_sharpe = pc.calculate_summary_statistics(portfolio_timeseries, return_as_range=True)
        spy_return, spy_volatility, spy_sharpe = pc.calculate_summary_statistics(spy_timeseries, return_as_range=False)
           
        summary_statistics = {
            "portfolio_esg_score": pc.calculate_esg_score(portfolio[['ticker','weight']].set_index('ticker', drop = True)),
            
            "portfolio_return_range": portfolio_return,
            
            "portfolio_volatility": portfolio_volatility,
            "portfolio_sharpe": portfolio_sharpe,
            
            "sp500_average_return": spy_return,
            "sp500_average_volatility": spy_volatility,
            "sp500_sharpe": spy_sharpe,
            
            "spy_max_dd": spy_max_dd,
            "portfolio_max_dd": portfolio_max_dd,
            
            # save memory by only returning / graphing every 5th datapoint
            "spy_timeseries": spy_timeseries[::5],
            "portfolio_timeseries": portfolio_timeseries[::5],
            "timeseries_dates": dates[::5],
            
            "portfolio_weighing_scheme": (
                "baseline_markowitz" if use_baseline_markowitz
                else "optimized_markowitz" if use_optimized_markowitz
                else "equal_weights"
            )
        }
        
        #package data into a dict of dicts for JsonResponse
    
        return JsonResponse({
            'sp500_compatibility': filter_results.set_index('ticker')['compatibility'].to_dict(),
            'portfolio': portfolio.set_index('ticker')['weight'].to_dict(),
            'summary_statistics': summary_statistics
            })

    return JsonResponse({"error": "Invalid request method."}, status=401)

@csrf_exempt
def update_weights(request):
    
    if request.method == "POST":
        
        # dict of request body:
        # tickers, risk_appetite, weighing_scheme
        update_request_dict = json.loads(request.body)
        
        # TODO update to process ticker and compatibility columns
        client_portfolio = pd.DataFrame(update_request_dict['client_portfolio'])
        
        # map risk appetite (0 - 0.2) to cash percent (50% - 10%)
        cash_percent = 0.5 - 2 * update_request_dict['risk_appetite']
        
        use_baseline_markowitz = (update_request_dict['weighing_scheme'] == 'Baseline Markowitz')
        use_optimized_markowitz = (update_request_dict['weighing_scheme'] == 'Optimized Markowitz')
        
        # print(client_portfolio.head())
        
        if(not client_portfolio.empty):
        
            # calculate best fit portfolio for the client
            ideal_portfolio_weights = pc.calculate_portfolio(client_portfolio[['ticker', 'compatibility']],
                                                             cash_percent=cash_percent,
                                                             use_baseline_markowitz=use_baseline_markowitz,
                                                             use_optimized_markowitz=use_optimized_markowitz,
                                                             return_summary_statistics=False)
            
            # set weights column to new weights
            client_portfolio['weight'] = ideal_portfolio_weights
            
            # reformat portfolio to single weight column indexed by ticker
            client_portfolio = client_portfolio[['ticker','weight']].set_index('ticker', drop = True)
            
            # read weights into dict format that is later returned to client
            weights_dict = client_portfolio['weight'].to_dict()
        
            # calculate updated portfolio timeseries
            portfolio_timeseries, dates, portfolio_max_dd = pc.portfolio_history(client_portfolio, 
                                                                                 include_spy=False)
            
            #calculate summary statistics   
            portfolio_return, portfolio_volatility, portfolio_sharpe = pc.calculate_summary_statistics(portfolio_timeseries, 
                                                                                                       return_as_range=True)
            portfolio_esg_score = pc.calculate_esg_score(client_portfolio)
            
        else:
            # empty portfolio case
            empty_portfolio = pd.DataFrame(columns=['ticker', 'weight'])
            empty_portfolio.set_index('ticker', inplace=True)
            
            # calculate timeseries which is only cash
            portfolio_timeseries, dates, portfolio_max_dd = pc.portfolio_history(empty_portfolio, include_spy=False)
            portfolio_esg_score = 0
            weights_dict = {}
        
        #calculate summary statistics   
        
        portfolio_return, portfolio_volatility, portfolio_sharpe = pc.calculate_summary_statistics(portfolio_timeseries, return_as_range=True)
        
        if(client_portfolio.empty):
            # overwrite calculation which overflows
            portfolio_sharpe = 0
        
             
        summary_statistics = {            
            "portfolio_return_range": portfolio_return,
            
            "portfolio_volatility": portfolio_volatility,
            "portfolio_sharpe": portfolio_sharpe,
            
            "portfolio_max_dd": portfolio_max_dd,
            
            "portfolio_esg_score": portfolio_esg_score,
            
            "portfolio_timeseries": portfolio_timeseries[::5],
            "timeseries_dates": dates[::5],
            
            "portfolio_weighing_scheme": (
                "baseline_markowitz" if use_baseline_markowitz
                else "optimized_markowitz" if use_optimized_markowitz
                else "equal_weights"
            )
        }
        
        # return format for updated_portfolio: {ticker: weight, ticker: weight etc}
        return JsonResponse({
            "updated_portfolio": weights_dict,
            "updated_summary_statistics": summary_statistics
        })
    
    return JsonResponse({"error": "Invalid request method."}, status=401)

@csrf_exempt
def update_risk(request):
    
    if request.method == "POST":
        
        # dict of request body:
        # tickers, risk_appetite, weighing_scheme
        update_request_dict = json.loads(request.body)
        
        # columns: ticker, weight
        client_portfolio = pd.DataFrame(update_request_dict['client_portfolio'])
        
        print(client_portfolio.head())
        
        # map risk appetite (0 - 0.2) to cash percent (50% - 10%)
        cash_percent = pc.map_risk_appetite_to_cash_percent(update_request_dict['risk_appetite'])
        
        current_equities_percent = client_portfolio['weight'].sum()
        scaling_factor = (1-cash_percent) / current_equities_percent
        
        client_portfolio['weight'] = client_portfolio['weight'] * scaling_factor
        
        portfolio_timeseries, dates, portfolio_max_dd = pc.portfolio_history(client_portfolio[['ticker','weight']].set_index('ticker', drop = True),
                                                                          include_spy=False)
        
        #calculate summary statistics   
        
        portfolio_return, portfolio_volatility, portfolio_sharpe = pc.calculate_summary_statistics(portfolio_timeseries, return_as_range=True)
        
             
        summary_statistics = {            
            "portfolio_return_range": portfolio_return,
            
            "portfolio_volatility": portfolio_volatility,
            "portfolio_sharpe": portfolio_sharpe,
            
            "portfolio_max_dd": portfolio_max_dd,
            
            "portfolio_timeseries": portfolio_timeseries[::5],
            "timeseries_dates": dates[::5],
        }
        
        # return format for updated_portfolio: {ticker: weight, ticker: weight etc}
        return JsonResponse({
            "updated_portfolio": client_portfolio[['ticker','weight']].set_index('ticker', drop = True)['weight'].to_dict(),
            "updated_summary_statistics": summary_statistics
        })
        
    return JsonResponse({"error": "Invalid request method."}, status=401)

@csrf_exempt
def simulate(request):
    if request.method == "POST":

        # build dict of client responses
        client_responses = json.loads(request.body)

        cash_percent = pc.map_risk_appetite_to_cash_percent(client_responses['riskAppetite'])

        # Run monte carlo simulation
        results = mc.run_monte_carlo_simulation(
            client_responses['portfolio'],
            cash_percent,
            client_responses['horizon'],
            client_responses['num_paths'],
            client_responses['advisor_fee'],
            client_responses['portfolio_weighing_scheme'],
            client_responses['rebalancing_rule'],
        )

        return JsonResponse(results, safe=False, status=200)
    
    return JsonResponse({"error": "Invalid request method."}, status=401)