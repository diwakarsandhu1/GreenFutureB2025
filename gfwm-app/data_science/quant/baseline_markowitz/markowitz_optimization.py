import pandas as pd
import numpy as np

import cvxopt as opt
from cvxopt import matrix, solvers, blas

#True to display progress in console
solvers.options["show_progress"] = True

# use adjusted to optimize and true to report (vol, return) points
def calculate_optimal_portfolios(true_mean_returns, adjusted_mean_returns, true_cov, adjusted_cov, 
                                 target_returns, annual_risk_free_rate = 0.02, bounds = None, calculate_best_fit = False):
    
    n = len(true_mean_returns)
    
    # convert to daily rate
    risk_free_rate = pow(annual_risk_free_rate+1, 1/365.0) - 1
    
    target_returns = target_returns[np.where(target_returns > risk_free_rate)]
    
    # initialize results dataframe
    empty_col = np.empty(len(target_returns))
    optimal_portfolios = pd.DataFrame({'target_return': target_returns,
                                   'annual_return': empty_col,
                                   'annual_volatility': empty_col,
                                   'weights': empty_col,
                                   'diversification': empty_col})

    
    # make sure these are floats    
    if bounds is None or len(bounds) != 2:
        bounds = [0.0, 2.0/n]
    elif(len(bounds) ==2):
        bounds[0] = float(bounds[0])
        bounds[1] = float(bounds[1])
    
    # minimize w * cov * w
    # subject to:
    # Gw <= h: (2n, n)(n, 1) <= (2n, 1)  
    #   -w <= -lower_bound (lower_bound <= w <= upper_bound)
    #   w <= upper_bound
    
    ''' example for n = 4
    -1  0   0   0       w1     =        -w1
    0   -1  0   0       w2              -w2
    0   0   -1  0       w3              -w3
    0   0   0  -1       w4              -w4
    1   0   0   0                       w1
    0   1   0   0                       w2
    0   0   1   0                       w3
    0   0   0   1                       w4
    '''
    
    # Aw = b: (2, n)(n, 1) = (2,1)
    # mean_returns * w = target_return
    # 1 * w = 1

    G = opt.matrix(0.0, (2 * n, n))
    for i in range(n):
        G[i, i] = -1.0
        G[n + i, i] = 1.0

    epsilon = 1e-8
    
    h = opt.matrix(-bounds[0] - epsilon, (2 * n, 1))
    for i in range(n):
        h[i + n] = bounds[1] + epsilon
    
    A = opt.matrix(1.0, (2, n))
    for i in range(n):
        A[0, i] = adjusted_mean_returns.iloc[i]

    # ensure matrices are in proper opt format
    
    true_mean_returns = matrix(true_mean_returns)
    adjusted_mean_returns = matrix(adjusted_mean_returns)
    
    true_cov = matrix(true_cov)
    adjusted_cov = matrix(adjusted_cov)
    
    target_returns = matrix(target_returns)
    
    q = matrix([0.0] * n)
    
    def solve_qp(target_return):
        # Run quadratic programming to minimize 1/2 * x^T cov * x with constraints
        solution = solvers.qp(P=adjusted_cov, q=q, G=G, h=h, A=A, b=matrix([target_return, 0.999999], (2, 1)))
    
        # Check solver status
        if solution['status'] == 'optimal':
            return solution['x']
    
    print(len(optimal_portfolios))
    optimal_portfolios['weights'] = optimal_portfolios['target_return'].map(solve_qp)
    
    # remove the rows where weights are none, ie no solution found
    optimal_portfolios.dropna(subset=['weights'], inplace=True)
    
    #print(len(optimal_portfolios))


    # Calculate annual return and annual volatility metrics based off of weights
    optimal_portfolios['annual_return'] = optimal_portfolios['weights'].map(
        lambda w: 252.0 * blas.dot(true_mean_returns, w)
    )
    optimal_portfolios['annual_volatility'] = optimal_portfolios['weights'].map(
        lambda w: np.sqrt(252.0 * blas.dot(w, true_cov*w))
    )
    
    optimal_portfolios['diversification'] = optimal_portfolios['weights'].map(
        lambda w: 1 - 1e4*np.sum((w - 1.0/n)**4)
    )
    
    print("Optimal Portfolios", optimal_portfolios)

    if(calculate_best_fit):
        # Calculate quadratic best fit
        # annual_return as independent variable, annual_volatility as dependent variable
        best_fit = np.polynomial.Polynomial.fit(
            optimal_portfolios['annual_return'], 
            optimal_portfolios['annual_volatility'], 2, domain=[0.0, 0.5])
        
        return optimal_portfolios, best_fit
    return optimal_portfolios


def montecarlo_random_portfolios(mean_returns, cov, bounds = None, iterations = 1e4):
    
    rng = np.random.default_rng()
    
    n = len(mean_returns)
    iterations = int(iterations)
    
    mean_returns = opt.matrix(mean_returns)
    cov = opt.matrix(cov)

    montecarlo_portfolios = pd.DataFrame(index=range(iterations))
    
    if(bounds is None):
        bounds = [0.0, n/4.0]

    # Generate random weights array and assign to 'random_weights' column
    montecarlo_portfolios['random_weights'] = list(
        rng.uniform(low=bounds[0], high=bounds[1], size=(iterations, n)))
    
    montecarlo_portfolios['random_weights'] = montecarlo_portfolios['random_weights'].map(
        lambda w: opt.matrix(np.array(w)/sum(w))
    )

    montecarlo_portfolios['annual_return'] = montecarlo_portfolios['random_weights'].map(
        lambda w: 252 * blas.dot(w, mean_returns)
    )

    montecarlo_portfolios['annual_volatility'] = montecarlo_portfolios['random_weights'].map(
        lambda w: np.sqrt(252.0 * blas.dot(w, cov*w))
    )
    
    return montecarlo_portfolios