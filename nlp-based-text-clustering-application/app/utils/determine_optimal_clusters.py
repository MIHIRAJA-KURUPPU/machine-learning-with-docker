
import numpy as np
from sklearn.cluster import KMeans

def determine_optimal_clusters(counts, max_clusters=10):
    """
    Use the elbow method to determine the optimal number of clusters.
    Returns the suggested number of clusters.
    """
    inertia_values = []
    for k in range(1, min(max_clusters + 1, counts.shape[0])):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(counts)
        inertia_values.append(kmeans.inertia_)
    
    # Simple elbow detection - find the point of maximum curvature
    diffs = np.diff(inertia_values)
    second_diffs = np.diff(diffs)
    
    # Return the point where the rate of change slows down the most
    # Default to 2 if we can't determine
    if len(second_diffs) > 0:
        return np.argmax(second_diffs) + 2
    return 2