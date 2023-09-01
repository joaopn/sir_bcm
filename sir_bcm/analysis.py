from sklearn.cluster import DBSCAN

def count_clusters(activity, epsilon):

    dbscan = DBSCAN(eps=epsilon, min_samples=2) 
    data = activity.reshape(-1, 1)

    clusters = dbscan.fit_predict(data)
    
    # Count number of isolated agents (no cluster)
    n_isolated = (clusters == -1).sum()
    
    # Count the number of unique clusters excluding noise (-1)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

    return n_clusters, n_isolated
