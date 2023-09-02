import pandas as pd
import random
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from sir_bcm.model_numba import run_model
from sir_bcm.analysis import count_clusters

@delayed
def simulate_and_analyze(n_users, t_max, epsilon, mu, trial, n_average=10, stop_tol=1e-6):
    seed = random.randint(0, 1e6)  # Generating a random seed
    try:
        opinion_mat, _, _ = run_model(n_users, t_max, epsilon, mu, n_average, stop_tol, seed=seed)
        t_stop = len(opinion_mat) - 1  # Calculating t_stop from the length of opinion_mat
        n_clusters, n_isolated = count_clusters(opinion_mat[-1], epsilon)  # Extracting both outputs
        return {
            "n_users": n_users,
            "epsilon": epsilon,
            "mu": mu,
            "trial": trial,
            "n_clusters": n_clusters,
            "n_isolated": n_isolated,  # Separate column for n_isolated
            "seed": seed,
            "t_stop": t_stop,
            "stop_tol": stop_tol  # Added stop_tol to the results
        }
    except ValueError:  # handle the exception raised by run_model when t_max is exceeded
        return {
            "n_users": n_users,
            "epsilon": epsilon,
            "mu": mu,
            "trial": trial,
            "n_clusters": None,
            "n_isolated": None,  # Handling cases where n_isolated might not be available due to the exception
            "seed": seed,
            "t_stop": None,
            "stop_tol": stop_tol  # Added stop_tol to the results
        }

def parallel_analysis(n_users, t_max, epsilons, mus, n_trials, n_cores=None, n_average=10, stop_tol=1e-6):
    if n_cores:
        cluster = LocalCluster(n_workers=n_cores, threads_per_worker=1)
        client = Client(cluster)
    
    results = []
    for epsilon in epsilons:
        for mu in mus:
            for trial in range(n_trials):
                results.append(simulate_and_analyze(n_users, t_max, epsilon, mu, trial, n_average, stop_tol))
    
    df_results = pd.DataFrame(compute(*results))
    
    if n_cores:
        client.close()
        cluster.close()
    
    return df_results
