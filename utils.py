import pandas as pd
import numpy as np
from typing import Tuple
import os
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


from cvxopt import matrix, solvers
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from scipy.stats import norm, rankdata
import cvxpy as cp



def load_dataset(
    csv_name: str = "2025 03 21 - EPFL Fin413 - Project Dataset Sendout.csv",
    folder : Tuple[str, None] = 'data'
) -> pd.DataFrame:
    
    if not csv_name.endswith('.csv'):
        csv_name += ".csv"
        
    path = os.getcwd()
    path = os.path.join(path, folder, csv_name)
        
    data = pd.read_csv(path, 
                    header=0, 
                    parse_dates=["Date"], 
                    dayfirst=True,
                    index_col="Date")


    data.columns = data.columns.str.replace("-", "_")


    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day

    return data

def compute_daily_returns(df: pd.DataFrame, log_returns=False, dropna=False) -> pd.DataFrame:
    """
    Compute daily returns for each column in the given price DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with daily prices.
    log_returns (bool): If True, compute log returns instead of percentage returns.
    dropna (bool): If True, drop rows with NaN values.

    Returns:
    pd.DataFrame: DataFrame with daily returns.
    """

    columns_to_drop = {"Day", "Month", "Year"}
    df.drop(columns=[column for column in columns_to_drop if column in df.columns], inplace=True)
    
    if log_returns:
        returns = np.log(df / df.shift(1))
    else:
        returns = df.pct_change()
    
    return returns.dropna() if dropna else returns.fillna(0)


def marchenkoPastur(X: pd.DataFrame) -> tuple:
    """
    Compute lower and higher values of the Marcenko-Pastur distribution.

    Args:
        X (pd.DataFrame): Random matrix of shape (T, N)
                                - T: number of samples 
                                - N: number of features.
                            We assume that the variance of the elements of X has been normalized to unity.

    Returns:
        tuple: Tuple containing lower and higher values.
    """
    
    T, N = X.shape
    q = N / float(T)

    lambda_min = (1 - np.sqrt(q))**2
    lambda_max = (1 + np.sqrt(q))**2

    return (lambda_min, lambda_max)


def clip(X: pd.DataFrame, alpha = None, return_covariance = True) -> pd.DataFrame:
    """
    Clip the eigenvalues of an empirical correlation matrix E.
    In this way, a new cleaned estimator E_clipped of the underlying correlation matrix is computed.
    To do so, the top [N * alpha] eigenvalues are kept and the remaining ones are skrinked.

    Args:
        X (pd.DataFrame): Random matrix of shape (T, N)
        alpha (float, optional): Value between 0 and 1.
                                It indicates the percentage of how many eigenvalues from the top have to be kept. Defaults to None.
        return_covariance (bool): Boolean which indicates if the output will be the clipped covarinace. Defaults to True.

    Returns:
        pd.DataFrame: Clipped matrix
    """
    
    T, N = X.shape
    
    X_std = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    
    E = np.corrcoef(X_std.T)
    
    eigvals, eigvecs = np.linalg.eigh(E)
    
    if alpha is None:
        (_, lambda_max) = marchenkoPastur(X_std)
        xi_clipped = np.where(eigvals >= lambda_max, eigvals, np.nan)
    else:
        xi_clipped = np.full(N, np.nan)
        threshold = int(np.ceil(alpha * N))
        if threshold > 0:
            xi_clipped[-threshold:] = eigvals[-threshold:]
            
    gamma = float(E.trace() - np.nansum(xi_clipped))
    gamma /= np.isnan(xi_clipped).sum()
    xi_clipped = np.where(np.isnan(xi_clipped), gamma, xi_clipped)

    E_clipped = np.zeros((N, N), dtype=float)
    for xi, eigvec in zip(xi_clipped, eigvecs):
        eigvec = eigvec.reshape(-1, 1)
        E_clipped += xi * eigvec.dot(eigvec.T)
        

    tmp = 1./np.sqrt(np.diag(E_clipped))
    E_clipped *= tmp
    E_clipped *= tmp.reshape(-1, 1)

    
    E_clipped = np.round(E_clipped, decimals=10)
    
    if return_covariance:
        std = np.sqrt(np.diag(np.cov(X.T)))
        E_clipped *= std
        E_clipped *= std.reshape(-1, 1)        
        return pd.DataFrame(E_clipped, index=X.columns, columns=X.columns)
    else:
        return pd.DataFrame(E_clipped, index=X.columns, columns=X.columns)
    
    
def compute_Euler_risk_contribution(weights: np.ndarray, Sigma: pd.DataFrame, percentage : bool = True) -> pd.Series:
    """
    Compute the Euler Risk Contribution for each asset.

    Args:
        weights (np.ndarray): Array containing the weight in the portfolio of each asset.
        Sigma (pd.DataFrame): Covariance matrix of the portfolio.
        percentage (bool, optional): If set to True, will return the percent contribution of each asset to the total risk.
                                        In this case, the sum of all contributions has to be equal to one.
                                        Defaults to True.

    Returns:
        pd.Series: Euler Risk Contributions.
    """
    
    # risk portfolio
    sigma_p = np.sqrt(weights.T @ Sigma @ weights).item()
    
    # marginal risk contribution
    mrc = Sigma @ weights
    
    # risk contribution
    rc = weights * mrc
    
    rc_normalized = rc / sigma_p
    
    if percentage:
        return rc_normalized / sigma_p
    else:
        return rc_normalized
    
    
def plot_Euler_risk_contribution(series: pd.Series, title: str) -> go.Figure:
    """
    Function to plot the Euler Risk Contributions using a bar chart.

    Args:
        series (pd.Series): Series containing the risk contributions
        title (str): Title of the Plot.

    Returns:
        go.Figure: Bar Chart Plot.
    """
    
    if np.round(sum(series), decimals=10) == 1:
        y = "Risk Contribution (%)"
        y_values = series.values * 100
    else:
        y = "Risk Contribution"
        y_values = series.values
        
    fig = px.bar(
        x=series.index,
        y=y_values, 
        labels={
            'x': "Assets",
            'y': y
        },
        title=title
    )
    fig.show()
    
def plot_eigenvalue_distribution(cov_matrix_raw, cov_matrix_clean, title):
    eigvals_raw = np.linalg.eigvalsh(cov_matrix_raw)
    eigvals_clean = np.linalg.eigvalsh(cov_matrix_clean)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(eigvals_raw, label='Raw Covariance', fill=True)
    sns.kdeplot(eigvals_clean, label='Clipped Covariance', fill=True)
    plt.title(f"Eigenvalue Distributions: {title}")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_3d_cov_matrix(cov_matrix, title):
    fig = go.Figure(data=[go.Surface(z=cov_matrix.values)])
    fig.update_layout(title=title, scene=dict(
        xaxis_title='Asset Index',
        yaxis_title='Asset Index',
        zaxis_title='Covariance Value'
    ))
    fig.show()
    
    
def compute_Herfindahl_Hirshmann_index(series : pd.Series) -> float:
    
    """
    Compute the Herfindahl index from the Euler Risk contributions.
    Source --> https://ijb.cyut.edu.tw/var/file/10/1010/img/861/V202-2.pdf 

    Returns:
        float: Herfindahl index
    """
    
    return np.sum(series.values ** 2)


def compute_erc_weights(cov_matrix: pd.DataFrame, c: float = 1e-1) -> pd.Series:
    """
    Compute equal risk contribution portfolio as described by Roncalli in the paper 'On the properties of equally-weighted risk contributions portfolios'.
    The optimization solved by this function can be found in Equation (7) of the paper.

    Args:
        cov_matrix (pd.DataFrame): Covariance matrix.
        c (float): Arbitrary costant. Ideally, it has to be small. Defaults to 1.

    Raises:
        ValueError: Error thrown in case the minimize algorithm does not converge.

    Returns:
        pd.Series: Equal risk contribution weights.
    """
    
    n_assets = cov_matrix.shape[0]

    def objective(y):
        return np.sqrt(y.T @ cov_matrix @ y)

    def log_sum_constraint(y):
        return np.sum(np.log(y)) - c

    # to avoid division by zero
    bounds = [(1e-8, None) for _ in range(n_assets)]

    initial_guess = np.random.dirichlet(np.ones(n_assets))

    constraints = [{
        'type': 'ineq',
         'fun': log_sum_constraint
    }]

    result = minimize(
        objective,
        x0=initial_guess,
        method='SLSQP', 
        bounds=bounds,
        tol=1e-6,
        constraints=constraints,
        options={'disp': False}
    )

    if result.success:
        y_star = result.x
        # Normalization
        weights = y_star / np.sum(y_star)
        return pd.Series(weights, index=cov_matrix.columns)
    else:
        raise ValueError(f"{result.message}")


def erc_objective(w, Sigma):
    Sigma_w = Sigma @ w
    rc = w * Sigma_w
    rc = np.array(rc)
    return np.sum((rc[:, None] - rc[None, :])**2)


def erc_portfolio_scipy(Sigma):
    n = Sigma.shape[0]
    
    # Initial guess: equally weighted
    w0 = np.ones(n) / n

    # Constraints
    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
        {'type': 'ineq', 'fun': lambda w: w}  # weights >= 0
    ]
    
    # Bounds for each weight: w_i >= 0, and w_max <= 10 * w_min
    bounds = [(0, 1) for _ in range(n)]  # basic bound [0,1]
    
    result = minimize(
        erc_objective, 
        w0, 
        args=(Sigma,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons,
        options={'disp': False, 'ftol': 1e-12, 'maxiter': 1000}
    )
    
    return result.x


def compute_condition_number(Sigma : pd.DataFrame) -> float:
    """
    Compute the condition number of a covariance matrix.

    Args:
        Sigma (pd.DataFrame): Covariance matrix.

    Returns:
        float: Condition number.
    """
    
    eig = np.linalg.eigvalsh(Sigma)
    min_eig, max_eig = np.min(eig), np.max(eig)
    return max_eig / min_eig



def calculate_effective_number_of_bets(weights, cov_matrix):
    """
    Calculate the Effective Number of Bets using Meucci's approach.
    
    Parameters:
    weights (np.array): Portfolio weights
    cov_matrix (np.array): Covariance matrix of returns
    
    Returns:
    tuple: (ENB, diversification_distribution)
    """

    weights = np.array(weights)
    cov_matrix = np.array(cov_matrix)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Transform portfolio weights to principal portfolio space
    transformed_weights = np.linalg.inv(eigenvectors) @ weights
    
    # portfolio variance
    portfolio_variance = weights.T @ cov_matrix @ weights
    
    # Calculate the diversific
    variance_concentrations = (transformed_weights**2) * eigenvalues
    diversification_distribution = variance_concentrations / portfolio_variance
    
    # Calculate Effective Number of Bets using Shannon entropy
    # Handle cases where p_i might be very close to zero
    nonzero_p = diversification_distribution[diversification_distribution > 1e-10]
    entropy = -np.sum(nonzero_p * np.log(nonzero_p))
    enb = np.exp(entropy)
    
    return enb, diversification_distribution


def maximize_diversification(cov_matrix: pd.DataFrame, long_only: bool=True, max_weight=None):
    """
    Compute the maximally diversified portfolio by maximizing the Effective Number of Bets.
    
    Parameters:
    -----------
    cov_matrix : numpy.ndarray or pandas.DataFrame
        Covariance matrix of asset returns
    long_only : bool, optional
        Whether to enforce long-only constraints, default is True
    max_weight : float, optional
        Maximum weight for any single asset (for concentration limits), default is None
        
    Returns:
    --------
    dict
        Dictionary containing optimal weights, ENB, and diversification distribution
    """
    n_assets = len(cov_matrix)
    
    # Convert to numpy array if pandas DataFrame
    if isinstance(cov_matrix, pd.DataFrame):
        asset_names = cov_matrix.index
        cov_matrix = cov_matrix.values
    else:
        asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Define objective function (negative ENB to minimize)
    def neg_enb(weights):
        enb, _ = calculate_effective_number_of_bets(weights, cov_matrix)
        return -enb  # Negative because we want to maximize ENB
    
    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Bounds
    if long_only:
        if max_weight is not None:
            bounds = tuple((0, max_weight) for _ in range(n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(n_assets))
    else:
        if max_weight is not None:
            bounds = tuple((-max_weight, max_weight) for _ in range(n_assets))
        else:
            bounds = tuple((None, None) for _ in range(n_assets))
    
    # Initial guess: equal weights
    initial_weights = np.ones(n_assets) / n_assets
    
    # Minimize negative ENB, ie maximize ENB
    result = minimize(
        neg_enb, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    # Get optimal weights
    optimal_weights = result['x']
    
    # Calculate final ENB and diversification distribution
    final_enb, div_dist = calculate_effective_number_of_bets(optimal_weights, cov_matrix)
    
    # Package results
    results = {
        'weights': pd.Series(optimal_weights, index=asset_names),
        'enb': final_enb,
        'diversification_ratio': final_enb / n_assets,
        'diversification_distribution': pd.Series(
            div_dist, 
            index=[f'PC{i+1}' for i in range(n_assets)]
        )
    }
    
    return results


def plot_diversification_distribution(div_dist, labels=None, title="Diversification Distribution"):
    """
    Plot the diversification distribution as an interactive bar chart using Plotly.
    
    Parameters:
    -----------
    div_dist : array-like
        The diversification distribution values to plot
    labels : array-like, optional
        Custom labels for the x-axis. If provided, must have the same length as div_dist
    title : str, default "Diversification Distribution"
        Title for the plot
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive Plotly figure
    """
    # Default x-values as integers if no labels provided
    if labels is None:
        x = np.arange(1, len(div_dist) + 1)
    else:
        # Validate that labels match the length of div_dist
        if len(labels) != len(div_dist):
            raise ValueError(f"Length of labels ({len(labels)}) must match length of div_dist ({len(div_dist)})")
        x = labels
    
    fig = go.Figure(data=[
        go.Bar(x=x, y=div_dist, marker=dict(color='steelblue'))
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='Principal Portfolios' if labels is None else 'Assets',
        yaxis_title='Contribution to Portfolio Risk',
        template='plotly_white',
        xaxis=dict(tickmode='linear' if labels is None else 'array', tickvals=x),
        yaxis=dict(gridcolor='lightgray'),
        bargap=0.2
    )
    
    return fig





def compute_mvp(cov_matrix):
    """
    Compute Minimum Variance Portfolio under:
      - Non-negativity constraints: wi ≥ 0
      - Full allocation constraint: sum(wi) = 1

    Parameters:
    - cov_matrix: pd.DataFrame
        Covariance matrix of asset returns (n x n)

    Returns:
    - pd.Series:
        Optimal portfolio weights (long-only, fully invested)
    """
    # Number of assets
    n_assets = cov_matrix.shape[0]
    
    # Define optimization variable (weights)
    w = cp.Variable(n_assets)

    # Define objective function: minimize variance
    Sigma = cov_matrix.values
    objective = cp.Minimize(cp.quad_form(w, Sigma))

    # Define constraints
    constraints = [
        cp.sum(w) == 1,  # Full investment (leverage = 1)
        w >= 0           # Non-negativity (no shorting)
    ]

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    # Check solver status
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed. Status: {problem.status}")

    # Convert to pandas Series with original asset names
    weights = pd.Series(w.value, index=cov_matrix.columns)

    # Post-process: ensure numerical stability (no small negative values)
    weights = weights.clip(lower=0)
    weights /= weights.sum()  # Re-normalize just in case

    return weights

def compute_erc(cov_matrix: pd.DataFrame, c: float = 0.1) -> pd.Series:
    """
    Compute Equal Risk Contribution (ERC) portfolio weights using SQP.

    Args:
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
        c (float): Log-sum constraint parameter to ensure diversification.

    Returns:
        pd.Series: ERC portfolio weights.
    """
    n_assets = cov_matrix.shape[0]

    def objective(y):
        return np.sqrt(y.T @ cov_matrix.values @ y)

    def log_sum_constraint(y):
        return np.sum(np.log(y)) - c

    # Bounds to ensure weights are positive
    bounds = [(1e-8, None) for _ in range(n_assets)]

    # Initial guess: equal weights
    initial_guess = np.repeat(1 / n_assets, n_assets)

    constraints = [{'type': 'ineq', 'fun': log_sum_constraint}]

    result = minimize(
        objective,
        x0=initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': False, 'maxiter': 1000, 'ftol': 1e-9}
    )

    if result.success:
        y_star = result.x
        weights = y_star / np.sum(y_star)
        return pd.Series(weights, index=cov_matrix.columns)
    else:
        raise ValueError(f"Optimization failed: {result.message}")
    

def compute_meucci_min_enb_portfolio(cov_matrix: pd.DataFrame) -> pd.Series:
    """
    Compute the Minimum ENB (Maximum Diversification Entropy) Portfolio à la Meucci (2009).

    Args:
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.

    Returns:
        pd.Series: Portfolio weights minimizing the Effective Number of Bets.
    """
    n = cov_matrix.shape[0]
    cov_matrix_values = cov_matrix.values
    
    # Initial guess (equal weights)
    w0 = np.ones(n) / n
    
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Bounds: weights between 0 and 1 (long-only portfolio)
    bounds = [(0, 1) for _ in range(n)]
    
    # Optimize to minimize the Effective Number of Bets
    result = minimize(
        enb_objective,
        w0,
        args=(cov_matrix_values,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    
    # Normalize weights (should already be normalized, but just in case)
    optimal_weights = result.x / np.sum(result.x)
    
    return pd.Series(optimal_weights, index=cov_matrix.index)

def enb_objective(w, cov_matrix):
    """
    Compute the objective function (negative of the effective number of bets portfolio entropy).

    Parameters:
        w (ndarray): Portfolio weights (n,)
        cov_matrix (ndarray): Covariance matrix Σ (n x n)
    
    Returns:
        float: Negative entropy of diversification distribution (to be maximized)
    """
    # Normalize weights to sum to 1
    w = w / np.sum(w)
    
    # Eigen-decomposition of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # Σ = e Λ² e.T
    eigvals = np.clip(eigvals, 1e-10, None)  # Ensure positive eigenvalues
    
    # Transform weights into uncorrelated space using e_t^T w
    x = eigvecs.T @ w  # Transform portfolio weights into uncorrelated space
    
    # Compute diversification distribution (squared normalized exposures)
    d = x**2 / np.sum(x**2)  # Diversification distribution
    
    # Compute entropy (with small epsilon to avoid log(0))
    epsilon = 1e-10
    d = np.clip(d, epsilon, 1 - epsilon)  # Avoid log(0)
    entropy = -np.sum(d * np.log(d))  # Entropy (diversification)
    
    # Return negative entropy for maximization (to minimize N_Ent)
    return np.exp(-entropy)  # Minimize the exponential of the negative entropy

def correl_distance(corr):
    """
    Convert a correlation matrix to a distance matrix for hierarchical clustering.

    Parameters:
    - corr: pd.DataFrame (correlation matrix)

    Returns:
    - np.ndarray: distance matrix
    """
    return np.sqrt(0.5 * (1 - corr))

def get_quasi_diag(link):
    """
    Quasi-diagonalize linkage matrix to get leaf ordering.

    Parameters:
    - link: np.ndarray (linkage matrix from scipy)

    Returns:
    - list: ordered indices of assets
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    
    return sort_ix.tolist()

def get_recursive_bisection_weights(cov, sorted_assets):
    """
    Recursively allocate portfolio weights based on hierarchical risk parity.

    Parameters:
    - cov: pd.DataFrame (covariance matrix)
    - sorted_assets: list of asset names in quasi-diagonal order

    Returns:
    - pd.Series: weights (long-only, sum to 1)
    """
    w = pd.Series(1.0, index=sorted_assets)
    cluster_items = [sorted_assets]

    while cluster_items:
        new_clusters = []
        for cluster in cluster_items:
            if len(cluster) <= 1:
                continue

            # Split cluster in half
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]

            def cluster_variance(subset):
                sub_cov = cov.loc[subset, subset]
                sub_w = w[subset]
                return np.dot(sub_w, sub_cov @ sub_w)

            var_left = cluster_variance(left)
            var_right = cluster_variance(right)
            total_var = var_left + var_right

            # Allocate based on inverse variance
            alpha = 0.5 if total_var == 0 else 1 - var_left / total_var
            w[left] *= alpha
            w[right] *= (1 - alpha)

            new_clusters.extend([left, right])

        cluster_items = new_clusters

    return w / w.sum()  # Ensure weights sum to 1

def compute_hrp(cov_matrix):
    corr = cov_matrix.corr()
    dist = correl_distance(corr)
    link = linkage(squareform(dist), method='single')
    sort_ix = get_quasi_diag(link)
    sorted_assets = cov_matrix.index[sort_ix].tolist()
    hrp_weights = get_recursive_bisection_weights(cov_matrix, sorted_assets)
    return hrp_weights.reindex(cov_matrix.columns).fillna(0.0)

def empirical_cdf(x):
    """Convert a 1D array into uniform [0,1] using ranks."""
    ranks = rankdata(x)
    return ranks / (len(ranks) + 1)

def copula_kendall_distance_matrix(log_returns_window: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a distance matrix using Kendall's tau implied by a Gaussian copula.

    Parameters:
        log_returns_window (pd.DataFrame): Window of log returns (T x N)

    Returns:
        pd.DataFrame: Copula-based distance matrix (N x N)
    """
    assets = log_returns_window.columns
    n = len(assets)
    tau_matrix = np.ones((n, n))  # Diagonal is 1

    for i in range(n):
        for j in range(i + 1, n):
            pair_data = log_returns_window[[assets[i], assets[j]]].dropna()

            u = empirical_cdf(pair_data.iloc[:, 0])
            v = empirical_cdf(pair_data.iloc[:, 1])

            z1 = norm.ppf(u)
            z2 = norm.ppf(v)

            rho = np.corrcoef(z1, z2)[0, 1]
            tau = (2 / np.pi) * np.arcsin(rho)

            tau_matrix[i, j] = tau
            tau_matrix[j, i] = tau

    # Convert Kendall's tau to distance
    dist_matrix = np.sqrt(0.5 * (1 - tau_matrix))
    return pd.DataFrame(dist_matrix, index=assets, columns=assets)

# hrp alternative function
def compute_hrp_alternative(log_returns_window: pd.DataFrame) -> pd.Series:
    """
    Compute Hierarchical Risk Parity weights using a copula-based distance metric.

    Parameters:
        log_returns_window (pd.DataFrame): Log return window (T x N)

    Returns:
        pd.Series: Portfolio weights (indexed by asset names)
    """
    # Step 1: Build distance matrix from copula-based Kendall's tau
    dist = copula_kendall_distance_matrix(log_returns_window)

    # Step 2: Hierarchical clustering
    link = linkage(squareform(dist), method='single')

    # Step 3: Get quasi-diagonal ordering
    sort_ix = get_quasi_diag(link)
    sorted_assets = log_returns_window.columns[sort_ix].tolist()

    # Step 4: Apply recursive bisection on the covariance matrix
    cov = log_returns_window.cov().loc[sorted_assets, sorted_assets]
    hrp_weights = get_recursive_bisection_weights(cov, sorted_assets)

    # Step 5: Reindex to original column order
    return hrp_weights.reindex(log_returns_window.columns).fillna(0.0)

def plot_comparison(hrp_standard, hrp_copula, date_label):
    """
    Plot a bar chart comparing standard HRP vs copula-based HRP weights.

    Args:
        hrp_standard (pd.Series): Weights from original HRP
        hrp_copula (pd.Series): Weights from alternative HRP using copula-based distance
        date_label (str): 'PP' or 'Tr'
    """
    x = np.arange(len(hrp_standard))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, hrp_standard, width, label='Standard HRP')
    ax.bar(x + width/2, hrp_copula, width, label='Alternative HRP')

    ax.set_ylabel('Weight')
    ax.set_title(f'HRP vs Alternative HRP Portfolio Weights ({date_label})')
    ax.set_xticks(x)
    ax.set_xticklabels(hrp_standard.index, rotation=90)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def compute_tsm_signals(log_returns: pd.DataFrame, lookback_days: int = 252) -> pd.Series:
    """
    Compute Time Series Momentum (TSM) signals: sign of past 12-month return for each asset.

    Parameters:
        log_returns (pd.DataFrame): Daily log returns.
        lookback_days (int): Lookback window in days (typically 252 for 1 year).

    Returns:
        pd.Series: +1 (long), -1 (short), or 0 (neutral) TSM signal per asset.
    """
    cumulative_returns = log_returns.tail(lookback_days).sum()
    signals = np.sign(cumulative_returns)
    return signals

def combine_tsm_hrpe(
    log_returns: pd.DataFrame,
    window_days: int = 252,
    method: str = 'HRPe'
) -> pd.Series:
    """
    Combine Time Series Momentum signals with HRPe structure.

    Parameters:
        log_returns (pd.DataFrame): Full time series of log returns.
        window_days (int): Lookback window for momentum and clustering.
        method (str): Must be 'HRPe'.

    Returns:
        pd.Series: Combined weights (in [-1, 1]).
    """
    # Step 1: Get last N days of data
    data_window = log_returns.tail(window_days)

    # Step 2: Compute HRPe weights (risk structure)
    dist = copula_kendall_distance_matrix(data_window)
    link = linkage(squareform(dist), method='single')
    sort_ix = get_quasi_diag(link)
    sorted_assets = data_window.columns[sort_ix].tolist()
    hrpe_weights = get_recursive_bisection_weights(data_window.cov(), sorted_assets)
    
    # Step 3: Compute TSM signals (+1, -1, 0)
    tsm_signals = compute_tsm_signals(log_returns, lookback_days=window_days)

    # Step 4: Combine — multiply structure by direction
    combined_weights = hrpe_weights * tsm_signals.loc[hrpe_weights.index]

    # Normalize to sum absolute weights = 1
    return combined_weights / combined_weights.abs().sum()

def plot_combined_weights(weights: pd.Series, title: str = "TSM + HRPe Portfolio Weights"):
    plt.figure(figsize=(14, 6))
    weights.sort_values().plot(kind='bar')
    plt.title(title)
    plt.ylabel("Weight")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def backtest_portfolio(log_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Compute cumulative return of a fixed-weight portfolio.

    Parameters:
        log_returns (pd.DataFrame): Daily log returns for all assets.
        weights (pd.Series): Portfolio weights (aligned with columns).

    Returns:
        pd.Series: Portfolio cumulative returns over time.
    """
    aligned_returns = log_returns[weights.index]
    port_log_returns = aligned_returns @ weights
    cumulative_returns = port_log_returns.cumsum().apply(np.exp)
    return cumulative_returns

def summarize_performance(returns, name):
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    return pd.Series({
        'Annual Return': ann_return,
        'Annual Volatility': ann_vol,
        'Sharpe Ratio': sharpe
    }, name=name)

