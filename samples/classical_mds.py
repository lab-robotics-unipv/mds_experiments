from __future__ import print_function
import operator

import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn import manifold
import matplotlib.pyplot as plt


#==========================================================================================================
# classical multidimensional scaling as described in https://en.wikipedia.org/wiki/Multidimensional_scaling
#==========================================================================================================

# number of dimensions of output
M = 2 

#         3m
#    -------------
#    |          /
#    |        /
# 4m |      /  5m
#    |    /
#    |  /
#    |/

# Classical MDS using eigen value decomposition
# --------------------------------------------------------------

# Set up the proximity and squared proximity matrices
prox_arr = np.array([[0, 5, 3], [5, 0, 4], [3, 4, 0]])
sqrd_prox_arr = prox_arr**2

# Apply double centering
sz = prox_arr.shape[0]
cent_arr = np.eye(sz) - np.ones(sz)/sz
B = -cent_arr.dot(sqrd_prox_arr).dot(cent_arr)/2

# Determine the M largest eigenvalues and corresponding eigenvectors
eig_vals, eig_vecs = np.linalg.eig(B)
eig_vals_vecs = zip(*sorted(zip(eig_vals, eig_vecs.T), key=operator.itemgetter(0), reverse=True)[:M])
eig_vals, eig_vecs = map(np.array, eig_vals_vecs)

# Configuration X of n points/coordinates that optimise the cost function
coords = eig_vecs.T.dot((np.eye(M)*eig_vals)**0.5)

# Compute euclidean distances
similairities = euclidean_distances(coords)
print(similairities)


# Classical MDS using SMACOF
# --------------------------------------------------------------

mds = manifold.MDS(metric=True, dissimilarity="precomputed")
coords2 = mds.fit(prox_arr).embedding_

# Compute euclidean distances
similairities2 = euclidean_distances(coords2)
print(similairities2)


# --------------------------------------------------------------

# Plot configurations, both should have slightly different orientations
plt.scatter(coords[:, 0], coords[:, 1], c='b')
plt.scatter(coords2[:, 0], coords2[:, 1], c='g')
plt.show()

# Compare the new proximity matrix with the original
np.testing.assert_allclose(similairities, prox_arr, 0.0001)
