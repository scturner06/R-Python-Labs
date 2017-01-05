# NOTES:
# 1) Don't know of a way to get individual sum of squares within cluster.
# 2) Omitted one line where they output object info.  Not very useful anyway.

# k-Means
import numpy as np
x = np.random.standard_normal(size = (100, 2))
x[0:25, 0] += 3
x[0:25, 1] -= 4

from sklearn.cluster import KMeans
km_out = KMeans(n_clusters = 2, n_init = 20, random_state = 1).fit(x)
km_out.labels_
km_out.inertia_
labels = km_out.labels_

import matplotlib.pyplot as plt
plt.scatter(x = x[:, 0], y = x[:, 1], c = labels)
plt.tit

# hiarchical clustering
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
hc_complete = hierarchy.linkage(pdist(x, 'euclidean'), method = 'complete', metric = 'euclidean')
plt.figure()
dn = hierarchy.dendrogram(hc_complete)
plt.show()

com_clust = hierarchy.fcluster(hc_complete, 2, criterion = 'maxclust')

x = np.random.standard_normal(size = (30, 3))
dd = pdist(1 - np.corrcoef(x))

# NCI60 Data Example
import pandas as pd

NCI60_data = pd.read_csv('../../data/NCI60_X.csv', index_col = 0)
NCI60_labs = pd.read_csv('../../data/NCI60_y.csv', index_col = 0)
