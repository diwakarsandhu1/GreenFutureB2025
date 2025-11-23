%Omega = xlsread('Returns_Covariance_Matrix_no_RZV.xlsx');
%mu = xlsread('Expected_Returns_no_RZV.xlsx');

Omega = readtable('sp500_covariance_matrix.csv');
data = readtable('sp500_daily_data.csv');

row_tickers = Omega.Properties.VariableNames(2:end); % Exclude the first column header
col_tickers = Omega{:, 1}; % First column as tickers

% Convert the table to a numeric matrix, excluding the first column
Omega = Omega{:, 2:end};

% get eigenvalues before adjustments to check positive semidefinite
[~, eigenvalues_matrix] = eig(Omega);
eigenvalues_before = diag(eigenvalues_matrix);

%{
threshold = 0.0008; % Set an appropriate threshold based on your data scale
[row, col] = find(abs(Omega) > threshold & ~eye(size(Omega)));
pairs = unique(sort([row, col], 2), 'rows');
tickers_to_remove = unique(pairs(:, 2)); % Choose to remove the second ticker in each pair

%remove those rows and columns from cov matrix
Omega(:, tickers_to_remove) = [];
Omega(tickers_to_remove, :) = [];

epsilon = 0.0003; % Small regularization term
Omega = Omega + epsilon * eye(size(Omega));
%}

%alternative: find nearest positive semi definite

% get eigenvalues after adjustments to confirm positive semidefinite
[~, eigenvalues_matrix] = eig(Omega);
eigenvalues_after = diag(eigenvalues_matrix);

% Group by ticker and calculate the mean of the 'log_return' column
mu = varfun(@mean, data, 'InputVariables', 'log_return', 'GroupingVariables', 'ticker');
%mu(tickers_to_remove,:) = [];

p = Portfolio('AssetMean',mu.mean_log_return, 'AssetCovar',Omega,'lb', 0,'budget', 1);
plotFrontier(p, 20);

p = setSolver(p, 'fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
        'SpecifyObjectiveGradient', true, 'SpecifyConstraintGradient', true, ...
        'ConstraintTolerance', 1.0e-8, 'OptimalityTolerance', 1.0e-8, 'StepTolerance', 1.0e-8); 

weights = estimateMaxSharpeRatio(p);       

te = 0.08;
p = setTrackingError(p,te,weights);

[risk, ret] = estimatePortMoments(p,weights);
hold on
plot(risk,ret,'*r');