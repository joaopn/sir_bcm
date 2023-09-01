import pandas as pd
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from sir_bcm.model import run_model
from sir_bcm.analysis import count_clusters

@delayed
def simulate_and_analyze(n_users, t_max, epsilon, mu, trial, n_average=10, stop_tol=1e-3):
    try:
        opinion_mat, _, _ = run_model(n_users, t_max, epsilon, mu, n_average, stop_tol)
        n_clusters = count_clusters(opinion_mat[-1], epsilon)
        return {"n_users": n_users, "epsilon": epsilon, "mu": mu, "trial": trial, "n_clusters": n_clusters}
    except ValueError:  # handle the exception raised by run_model when t_max is exceeded
        return {"n_users": n_users, "epsilon": epsilon, "mu": mu, "trial": trial, "n_clusters": None}

def parallel_analysis(n_users, t_max, epsilons, mus, n_trials, n_cores=None, n_average=10, stop_tol=1e-3):
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