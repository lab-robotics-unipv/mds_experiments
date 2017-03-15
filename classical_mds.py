from __future__ import print_function
import operator

import numpy as np
from sklearn.metrics import euclidean_distances

#
# classical multidimensional scaling as described in https://en.wikipedia.org/wiki/Multidimensional_scaling
#

M = 2 # number of dimensions of output

#       5m
#  ................
#   .           .
#    .        .
# 3m  .     .  4m
#      .  .
#       .


# Set up the squared proximity matrix
prox_arr = np.array([[0, 5, 3], [5, 0, 4], [3, 4, 0]])
sqrd_prox_arr = prox_arr**2

# Apply double centering
sz = prox_arr.shape[0]
cent_arr = np.eye(sz) - np.ones(sz)/sz
B = -cent_arr.dot(sqrd_prox_arr).dot(cent_arr)/2

# Determine the m largest eigenvalues and corresponding eigenvectors
e_vals, e_vecs = np.linalg.eig(B)
e_vals, e_vecs = map(np.array, zip(*sorted(zip(e_vals, e_vecs.T), key=operator.itemgetter(0), reverse=True)[:M]))

# configuration X of n points/coordinates that optimise the cost function
coords = e_vecs.T.dot((np.eye(M)*e_vals)**0.5)
print(coords)

# compute euclidean distances
similairities = euclidean_distances(coords)
print(similairities)

np.testing.assert_allclose(similairities, prox_arr)
