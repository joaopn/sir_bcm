import numpy as np
from sklearn.cluster import DBSCAN

def count_clusters(activity, epsilon):

    dbscan = DBSCAN(eps=epsilon, min_samples=2) 
    data = activity.reshape(-1, 1)

    clusters = dbscan.fit_predict(data)
    
    # Count number of isolated agents (no cluster)
    n_isolated = (clusters == -1).sum()
    
    # Count the number of unique clusters excluding noise (-1)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

    # Calculate mean value of each cluster excluding noise (-1)
    unique_clusters = [x for x in np.unique(clusters) if x != -1]
    cluster_means = [data[clusters == cluster].mean() for cluster in unique_clusters]

    return n_clusters, n_isolated, cluster_means
