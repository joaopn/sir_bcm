import numpy as np
import pandas as pd


def initialize_users(n_users):
    opinions = np.random.uniform(0, 1, n_users)
    influences = np.random.uniform(0, 1, n_users)
    return opinions, influences

# process data
def extract(csvfile):#csvfile is the address of the csv
    df=pd.read_csv(csvfile)
    return df['count'].tolist(),df['value'].tolist()

# generate influence vector
def influence_generator(count,value,n_users):
    divider = sum(count)
    p=[num / divider for num in count]
    # divider = sum(value)
    # v=[num / divider for num in value]
    # influences=np.random.choice(v, p=p,size=n_users)
    influences=np.random.choice(value, p=p,size=n_users)
    print(influences)
    # influences=influences/np.sum(influences)
    # print(influences)
    return influences

# initialize users with empirical data 
def initialize_users_activity_driven(n_users):
    opinions = np.random.uniform(0, 1, n_users)
    # influences = influence_generator(count,value)
    return opinions

# select influencer based on weights
def select_influencer_activity_driven(influences):
    divider = sum(influences)
    p=[num / divider for num in influences]
    influencer = np.random.choice(np.arange(len(influences)), p=p)
    # influencer = np.random.choice(np.arange(len(influences)), p=influences)
    return influencer

def select_influencer(influences):
    normalized_influences = influences / np.sum(influences)
    return np.random.choice(np.arange(len(influences)), p=normalized_influences)

def update_opinion(opinions, selected_user, influencer, epsilon, mu):
    difference = opinions[influencer] - opinions[selected_user]
    if abs(difference) <= epsilon:
        opinions[selected_user] += difference*mu

def has_stabilized(opinion_matrix, t, n_average, stop_tol):
    if t < n_average:
        return False
    variances = np.var(opinion_matrix[t-n_average:t], axis=0)
    max_variance = np.max(variances)
    return max_variance < stop_tol

def run_model(n_users, t_max, epsilon, mu, n_average=10, stop_tol=1e-6):
    """Run the model for n_users, t_max, epsilon, and mu.

    Parameters
    ----------
    n_users : int
        Number of users in the model.
    t_max : int
        Maximum number of timesteps to run the model for.
    epsilon : float
        Threshold for opinion difference between users.
    mu : float
        Learning rate.
    n_average : int, optional
        Number of timesteps to average over when checking for stabilization.
        Defaults to 10.
    stop_tol : float, optional
        Maximum variance in opinions to consider the model stabilized.
        Defaults to 1e-6.
    
    Returns
    -------
    opinion_matrix : np.ndarray
        Matrix of opinions over time.
    influences : np.ndarray
        Array of influences for each user.
    t : int
        Stopping timestep.
    """
    
    opinions, influences = initialize_users(n_users)
    opinion_matrix = np.zeros((t_max, n_users))
    opinion_matrix[0] = opinions.copy()

    for t in range(1, t_max):
        for user in range(n_users):
            influencer = select_influencer(influences)
            update_opinion(opinions, user, influencer, epsilon, mu)
        opinion_matrix[t] = opinions.copy()
        
        if has_stabilized(opinion_matrix, t, n_average, stop_tol):
            return opinion_matrix[:t+1], influences, t  # Return the opinions up to timestep t, influences, and stopping timestep
        
    raise ValueError(f"Model did not stabilize within {t_max} timesteps!")

def run_model_activity_driven(n_users, t_max, epsilon, mu, csvfile, n_average=10, stop_tol=1e-6):
    
    opinions = initialize_users_activity_driven(n_users)
    count,value = extract(csvfile)
    influences = influence_generator(count,value,n_users)
    opinion_matrix = np.zeros((t_max, n_users))
    opinion_matrix[0] = opinions.copy()

    for t in range(1, t_max):
        for user in range(n_users):
            influencer = select_influencer_activity_driven(influences)
            update_opinion(opinions, user, influencer, epsilon, mu)
        opinion_matrix[t] = opinions.copy()
        
        if has_stabilized(opinion_matrix, t, n_average, stop_tol):
            return opinion_matrix[:t+1], influences, t  # Return the opinions up to timestep t, influences, and stopping timestep
        
    raise ValueError(f"Model did not stabilize within {t_max} timesteps!")