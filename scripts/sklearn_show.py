import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

n_samples = 1500
# X, _ = datasets.make_moons(n_samples,noise = 0.1,random_state=1)
# X, _ = datasets.make_blobs(n_samples=n_samples, random_state=8)
# X, _ = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)

x1 = X[:, 0]
y1 = X[:, 1]
plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(x1, y1, s=10, cmap='viridis')

# normalize dataset for easier parameter selection
# X = StandardScaler().fit_transform(X)

# algorithm = mixture.GaussianMixture(n_components=3, covariance_type='full')
# algorithm.fit(X)

algorithm = cluster.SpectralClustering(
        n_clusters=3, eigen_solver='arpack',
        affinity="nearest_neighbors")
algorithm.fit(X)

if hasattr(algorithm, 'labels_'):
    y_pred = algorithm.labels_.astype(np.int)
else:
    y_pred = algorithm.predict(X)

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                    '#f781bf', '#a65628', '#984ea3',
                                    '#999999', '#e41a1c', '#dede00']),
                            int(max(y_pred) + 1))))
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

plt.show()