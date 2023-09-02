import pandas as pd
import random
from dask import delayed, compute
from dask.distributed import Client, TimeoutError
from sir_bcm.model_numba import run_model
from sir_bcm.analysis import count_clusters

@delayed
def simulate_and_analyze(n_users, t_max, epsilon, mu, trial, n_average=10, stop_tol=1e-6):
    seed = random.randint(0, 1e6)  # Generating a random seed
    try:
        opinion_mat, _, _ = run_model(n_users, t_max, epsilon, mu, n_average, stop_tol, seed=seed)
        t_stop = len(opinion_mat) - 1  # Calculating t_stop from the length of opinion_mat
        n_clusters, n_isolated, cluster_means = count_clusters(opinion_mat[-1], epsilon)  # Extracting both outputs
        return {
            "n_users": n_users,
            "epsilon": epsilon,
            "mu": mu,
            "trial": trial,
            "n_clusters": n_clusters,
            "n_isolated": n_isolated,  # Separate column for n_isolated
            "cluster_means": cluster_means,
            "seed": seed,
            "t_stop": t_stop,
            "stop_tol": stop_tol  # Added stop_tol to the results
        }
    except ValueError:  # handle exceptions
        return {
            "n_users": n_users,
            "epsilon": epsilon,
            "mu": mu,
            "trial": trial,
            "n_clusters": None,
            "n_isolated": None,  # Handling cases where n_isolated might not be available due to the exception
            "cluster_means": None, 
            "seed": seed,
            "t_stop": None,
            "stop_tol": stop_tol  # Added stop_tol to the results
        }

def parallel_analysis(n_users, t_max, epsilons, mus, n_trials, cluster_ip, n_average=10, stop_tol=1e-6):
    scheduler_address = f"tcp://{cluster_ip}:8786"
    try:
        # Attempt to connect to the cluster
        client = Client(scheduler_address, timeout=5)  # Timeout in 5 seconds if no connection
    except TimeoutError:
        raise RuntimeError(f"Failed to connect to the Dask cluster at {scheduler_address}. Ensure the cluster is up and running.")

    results = []
    for epsilon in epsilons:
        for mu in mus:
            for trial in range(n_trials):
                results.append(simulate_and_analyze(n_users, t_max, epsilon, mu, trial, n_average, stop_tol))
    
    df_results = pd.DataFrame(compute(*results))
    
    client.close()
    
    return df_results

    