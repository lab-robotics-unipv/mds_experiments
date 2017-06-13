from __future__ import print_function
import datetime

from itertools import cycle
import numpy as np
from sklearn.metrics import euclidean_distances

from core import mds, motion



def generate_static_nodes(config, algorithm, add_noise=None, filter_noise=False, step=0, show_trail=True):
	NO_OF_TAGS, NO_OF_ANCHORS = config.no_of_tags, config.no_of_anchors
	if filter_noise and not add_noise:
		raise ValueError("can't filter when there is no noise")
	curr_coords = config.generate_points()
	prox_arr = euclidean_distances(curr_coords)
	if getattr(config, 'missingdata', None):
		prox_arr[-NO_OF_TAGS:, -NO_OF_TAGS:] = 0

	# add noise to between-sets proximities
	mu = config.mu
	sigma = config.sigma
	if sigma > 0:
		noise = np.random.normal(mu, sigma, size=(NO_OF_ANCHORS, NO_OF_TAGS)) 
		prox_arr[:NO_OF_ANCHORS, -NO_OF_TAGS:] += noise
		prox_arr[-NO_OF_TAGS:, :NO_OF_ANCHORS] += noise.T

		if getattr(config, 'missingdata') is None: 
			# add noise to within-sets data when available
			noise = np.random.normal(mu, sigma, size=(NO_OF_TAGS, NO_OF_TAGS))
			prox_arr[-NO_OF_TAGS:, -NO_OF_TAGS:] += (noise+noise.T)/2.0
			# self distance should remain zero
			prox_arr[np.arange(len(prox_arr)), np.arange(len(prox_arr))] = 0

	#TODO: attenuation or multipath would mean a longer range as seen from the mobile node
	slc = slice(None)
	if filter_noise:
		# drop/filter first component in computation of configuration 
		slc = slice(1, None)
	results = estimate_coords(config, algorithm, prox_arr[slc, slc])
	coords = results.embedding_

	# Use anchor coordinates to compute rotation and translation matrices
	Rotation, t, s = best_similarity_transform(coords[:-NO_OF_TAGS, :].T, curr_coords[slc.start:-NO_OF_TAGS, :].T)
	coords = (Rotation.dot((coords/s).T) + t).T
	configs = curr_coords, coords
	if show_trail:
		last_n_coords = results.last_n_embeddings
		for i in range(len(last_n_coords)):
			last_n_coords[i] = (Rotation.dot((last_n_coords[i]/s).T) + t).T
		configs += (last_n_coords,)

	return configs


def generate_dynamic_nodes(config, algorithm, add_noise, filter_noise, no_of_trans):
	# change coordinate of mobile nodes using transition step value
	for i in cycle([0, 1, 2, 3]):
		step = (i%no_of_trans)/float(30)
		yield generate_static_nodes(config, algorithm, add_noise=add_noise, filter_noise=filter_noise, step=step)


def computeRSSI(curr_coords):
	dist = euclidean_distances(curr_coords)
	dist[dist==0] = 0.7
	dist[:-NO_OF_TAGS, :-NO_OF_TAGS] = -35 - (2*10)*np.log10(dist[:-NO_OF_TAGS, :-NO_OF_TAGS])
	# use varying alpha values for each anchor as seen from the tag
	dist[-NO_OF_TAGS, :] = dist[:, -NO_OF_TAGS] = -35 - (ALPHA*10)*np.log10(dist[-NO_OF_TAGS, :])
	#print(dist)
	return dist


def estimate_coords(config, algorithm, prox_arr):
	seed = np.random.RandomState(seed=3)
	mds_ = mds.MDS(config, algorithm, n_components=2, max_iter=500, eps=1e-14, random_state=seed, n_jobs=1, dissimilarity="precomputed", metric=True, verbose=0) # n_init=1
	results = mds_.fit(prox_arr)
	return results


def best_similarity_transform(X, Y):
	K = X.shape[0]
	Xm = X.mean(axis=1).reshape(K, 1)
	Ym = Y.mean(axis=1).reshape(K, 1)
	X = X - Xm
	Y = Y - Ym
	normX = np.linalg.norm(X)
	normY = np.linalg.norm(Y)
	X /= normX
	Y /= normY

	YX = np.dot(Y, X.T)
	U, S, V = np.linalg.svd(YX)
	R = np.dot(U, V)
	t = Ym - np.dot(R, Xm) 
	s = S.sum() * normX / normY
	return R, t, s