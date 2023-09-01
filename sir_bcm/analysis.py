from sklearn.cluster import DBSCAN

def count_clusters(activity, epsilon):

    dbscan = DBSCAN(eps=epsilon, min_samples=2) 
    data = activity.reshape(-1, 1)

    clusters = dbscan.fit_predict(data)
    return len(set(clusters)) - (1 if -1 in clusters else 0)
