from numba import jit
import numpy as np

@jit(nopython=True)
def custom_choice(probs):
    """Custom function to emulate np.random.choice based on probabilities"""
    cumulative = 0.0
    rand_val = np.random.uniform(0, 1)
    for idx, prob in enumerate(probs):
        cumulative += prob
        if rand_val < cumulative:
            return idx
    return len(probs) - 1  # Return the last index as a fallback

@jit(nopython=True)
def custom_mean(matrix, axis=0):
    """Compute the mean along a specific axis."""
    sum_vals = np.sum(matrix, axis=axis)
    if axis == 0:
        return sum_vals / matrix.shape[0]
    else:
        return sum_vals / matrix.shape[1]

@jit(nopython=True)
def custom_var(matrix, axis=0):
    """Compute the variance along a specific axis."""
    mean_vals = custom_mean(matrix, axis=axis)
    deviations = matrix - mean_vals
    squared_deviations = deviations ** 2
    variances = custom_mean(squared_deviations, axis=axis)
    return variances

@jit(nopython=True)
def initialize_users(n_users, seed=None):
    if seed is not None:
        np.random.seed(seed)
    opinions = np.random.rand(n_users)
    influences = np.random.rand(n_users)
    return opinions, influences

@jit(nopython=True)
def select_influencer(influences):
    normalized_influences = influences / np.sum(influences)
    return custom_choice(normalized_influences)

@jit(nopython=True)
def update_opinion(opinions, selected_user, influencer, epsilon, mu):
    difference = opinions[influencer] - opinions[selected_user]
    if abs(difference) <= epsilon:
        opinions[selected_user] += difference*mu

@jit(nopython=True)
def has_stabilized(opinion_matrix, t, n_average, stop_tol):
    if t < n_average:
        return False
    variances = custom_var(opinion_matrix[t-n_average:t], axis=0)
    max_variance = np.max(variances)
    return max_variance < stop_tol

@jit(nopython=True)
def run_model(n_users, t_max, epsilon, mu, n_average=10, stop_tol=1e-6, seed=None):
    opinions, influences = initialize_users(n_users, seed)
    opinion_matrix = np.zeros((t_max, n_users))
    opinion_matrix[0] = opinions.copy()

    for t in range(1, t_max):
        for user in range(n_users):
            influencer = select_influencer(influences)
            update_opinion(opinions, user, influencer, epsilon, mu)
        opinion_matrix[t] = opinions.copy()
        
        if has_stabilized(opinion_matrix, t, n_average, stop_tol):
            return opinion_matrix[:t+1], influences, t
        
    raise ValueError(f"Model did not stabilize within {t_max} timesteps!")