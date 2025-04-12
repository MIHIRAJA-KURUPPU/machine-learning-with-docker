import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from io import BytesIO

def create_elbow_plot(counts, max_clusters=10):
    """Create an elbow plot for determining optimal number of clusters"""
    inertia_values = []
    k_values = range(1, min(max_clusters + 1, counts.shape[0]))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(counts)
        inertia_values.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia_values, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf
