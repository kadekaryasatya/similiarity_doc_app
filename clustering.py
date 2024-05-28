from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_clustering(similarity_matrix, num_clusters):
    if num_clusters == 2:
        # Jika jumlah kluster adalah 2, tetapkan kluster secara langsung
        labels = [0] * len(similarity_matrix)
        silhouette_avg = None  # Tidak ada silhouette score karena hanya satu kluster
    else:
        # Jika jumlah kluster bukan 2, gunakan algoritma KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(similarity_matrix)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(similarity_matrix, labels)
    return silhouette_avg, labels
