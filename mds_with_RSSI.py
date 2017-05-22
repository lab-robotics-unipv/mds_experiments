from __future__ import print_function
import operator
import datetime

from itertools import count, cycle, tee, islice
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn import manifold
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import animation

import mds
import motion
import evaluation

from config import *
import config


ALGORITHMS = ('_smacof_single', '_smacof_with_anchors_single', '_smacof_with_distance_recovery_single')
LABELS = dict(zip(ALGORITHMS, ['Classical MDS', 'Anchored MDS', 'MDS-RFID']))

def compute_config(config, anc, algorithm, add_noise=None, filter_noise=False, step=0, show_trail=True):
	NO_OF_TAGS, NO_OF_ANCHORS = config.NO_OF_TAGS, config.NO_OF_ANCHORS
	if filter_noise and not add_noise:
		raise ValueError("can't filter when there is no noise")
	coords_init = generate_points(anc, NO_OF_TAGS)
	
	tag_prev_pos = tag_curr_pos = prev_time = None
	time_duration = 0

	curr_coords = coords_init.copy()
	
	#move first tag
	curr_coords[-2, 0] -= step
	curr_coords[-2, 1] += step
	#move second tag
	curr_coords[-1, 0] += 0.7*step 

	#move third tag
	#curr_coords[-1, 0] -= 0.4*step 
	#curr_coords[-1, 1] -= step 

	# set up dissimilarity matrix containing RSSI

	#prox_arr = computeRSSI(curr_coords)
	#prox_arr = 10 ** ((-35 - prox_arr)/(10*np.mean(ALPHA)))
	#print(prox_arr)
	#prox_arr -= prox_arr.mean()
	prox_arr = euclidean_distances(curr_coords)
	if getattr(config, 'MISSINGDATA') is not None:
		prox_arr[-NO_OF_TAGS:, -NO_OF_TAGS:] = 0

	#
	# scale range values 
	# prox_arr *= 1.1
	#prox_arr[prox_arr==np.inf] = 0

	
	if add_noise:
		# add noise to between-sets proximities
		mu = add_noise.get('mu', 0)
		sigma = add_noise.get('sigma', 0)
		if sigma !=0:
			noise = np.random.normal(mu, sigma, size=(NO_OF_ANCHORS, NO_OF_TAGS)) 
			prox_arr[:NO_OF_ANCHORS, -NO_OF_TAGS:] += noise
			prox_arr[-NO_OF_TAGS:, :NO_OF_ANCHORS] += noise.T

			if getattr(config, 'MISSINGDATA') is None: 
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


	if prev_time:
		time_duration = (datetime.datetime.now() - prev_time).total_seconds()
	tag_curr_pos = coords[-1]
	if tag_prev_pos is not None and tag_curr_pos is not None and time_duration:
		pass #print(motion.compute_velocity(tag_prev_pos, tag_curr_pos, time_duration))
	tag_prev_pos = tag_curr_pos
	prev_time = datetime.datetime.now()

	return configs


def generate_data(config, anc, algorithm, add_noise, filter_noise):
	# change coordinate of mobile nodes using transition step value
	for i in cycle([0]):
		step = (i%NO_TRANS)/float(30)
		yield compute_config(config, anc, algorithm, add_noise=add_noise, filter_noise=filter_noise, step=step)


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


class NodeAnimation(object):

	def __init__(self, data, show_trail=False):
		self.fig = plt.figure()
		self.data = data
		self.show_trail = show_trail

	def init_plot(self):
		real_coords, mds_coords, *others = next(self.data)
		self.anchors_scat = plt.scatter(real_coords[:NO_OF_ANCHORS, 0], real_coords[:NO_OF_ANCHORS, 1], color='blue', s=100, lw=1, label='Anchors positions', marker='o')
		self.real_scat = plt.scatter(real_coords[NO_OF_ANCHORS:, 0], real_coords[NO_OF_ANCHORS:, 1], color='blue', s=100, lw=1, label='Tag positions', marker='o', facecolors='none')
		self.mds_scat = plt.scatter(mds_coords[:, 0], mds_coords[:, 1], color='red', s=150, lw=2, label='Estimated positions', marker='+')
		scatterplots = self.real_scat, self.mds_scat
		if self.show_trail:
			last_n_mds_coords = others[0][2:]
			self.last_n_scat = plt.scatter(last_n_mds_coords[:, :, 0], last_n_mds_coords[:, :, 1], label='MDS iterations', color='magenta', s=0.5)
			scatterplots += (self.last_n_scat,)
		plt.legend(loc='best', scatterpoints=1, fontsize=11)
		return scatterplots
		
	def update_plot(self, _):
		real_coords, mds_coords, *others = next(self.data)
		self.real_scat.set_offsets(real_coords)
		self.mds_scat.set_offsets(mds_coords)
		scatterplots = self.real_scat, self.mds_scat
		if self.show_trail:
			last_n_mds_coords = others[0][2:]
			self.last_n_scat.set_offsets(last_n_mds_coords)
			scatterplots += (self.last_n_scat,)
		return scatterplots

	def draw_plot(self, save_to_file=False):
		anim = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, interval=300)
		

		# matplotlib bug? Not calling plt.show in the same scope as anim gives a blank figure
		axes = plt.gca()
		
		axes.set_xlabel('coordinate x [m]')
		axes.set_ylabel('coordinate y [m]')
		axes.set_xlim([-5, 35])
		axes.set_ylim([-5, 29])
		if save_to_file:
			anim.save('mds_animation_filtering.mp4', extra_args=['-vcodec', 'libx264'])
		
		plt.show()





if __name__ == '__main__':
	anc = 4
	config.NO_OF_ANCHORS = anc
	config.NO_OF_TAGS = 7
	config.ANCHORS = config.ANCHORS_[:anc]
	# config.MISSINGDATA = 'None'
	# noise = dict(mu=0, sigma=1)
	# for_plotting, for_evaluation = tee(generate_data(config, anc, '_smacof_with_anchors_single', add_noise=noise, filter_noise=False))
	# anim = NodeAnimation(for_plotting,show_trail=False)
	# anim.draw_plot()




	# noise = {'mu': 0}
	# plot_hanldes = []
	# plot_labels = []
	# for algorithm in ALGORITHMS:
	# 	errors = []
	# 	sigmas = np.linspace(0, 4, 40)
	# 	for sigma in sigmas:
	# 		noise['sigma'] = sigma

	# 		for_evaluation = generate_data(config, algorithm, filter_noise=False, add_noise=noise)

	# 		# generate rmse from coordinates, remembering to not pass last_n_coords to function
	# 		error = [evaluation.rmse(*coords[:2]) for coords in islice(for_evaluation, None, 500)]
	# 		errors.append(error)
	# 	errors = np.array(errors)
	# 	first_q, median, third_q = evaluation.first_third_quartile_and_median(errors)
	# 	#np.save()
	# 	handles = evaluation.plot_rmse_vs_sigma(first_q, median, third_q, x_axis=sigmas, algorithm=algorithm)
	# 	plot_hanldes.extend(handles)
	# 	plot_labels.extend([LABELS[algorithm], 'IQR boundaries'])
	# plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_hanldes)
	# plt.show()




	# noise = {'mu': 0, 'sigma': 3}
	# plot_hanldes = []
	# plot_labels = []
	# no_of_anchors = range(3,9)
	# for algorithm in ALGORITHMS:
	# 	errors = []
		
	# 	for anc in no_of_anchors:

	# 		config.NO_OF_ANCHORS = anc
	# 		config.ANCHORS = config.ANCHORS_[:anc]
	# 		for_evaluation = generate_data(config, anc, algorithm, filter_noise=False, add_noise=noise)

	# 		# generate rmse from coordinates, remembering to not pass last_n_coords to function
	# 		error = [evaluation.rmse(*coords[:2]) for coords in islice(for_evaluation, None, 100)]
	# 		errors.append(error)
	# 	errors = np.array(errors)
	# 	first_q, median, third_q = evaluation.first_third_quartile_and_median(errors)
	# 	#np.save()
	# 	handles = evaluation.plot_rmse_vs_anchors(first_q, median, third_q, x_axis=no_of_anchors, algorithm=algorithm)
	# 	plot_hanldes.extend(handles)
	# 	plot_labels.extend([LABELS[algorithm], 'IQR boundaries'])
	# plt.legend(loc='upper right', fontsize=12, labels=plot_labels, handles=plot_hanldes)
	# plt.show()




	# noise = {'mu': 0, 'sigma': 3}
	# plot_hanldes = []
	# plot_labels = []
	# no_of_anchors = range(3,9)
	# for algorithm in ALGORITHMS:
	# 	n_iters = []
		
	# 	for anc in no_of_anchors:

	# 		config.NO_OF_ANCHORS = anc
	# 		config.ANCHORS = config.ANCHORS_[:anc]
	# 		for_evaluation = generate_data(config, anc, algorithm, filter_noise=False, add_noise=noise)

	# 		n_iter = np.array(list(islice(for_evaluation, None, 100))).mean()
	# 		n_iters.append(n_iter)
	# 	n_iters = np.array(n_iters)
	# 	np.savetxt('{}.txt'.format(algorithm), n_iters)




	ALGORITHMS = ['MDS-A, no missing data', 'modified MDS-A, missing inter-tag data']
	noise = {'mu': 0}
	plot_hanldes = []
	plot_labels = []
	for i in range(2):
		if i == 0:
			config.MISSINGDATA = None
		else:
			config.MISSINGDATA = 'yes'
		print(config.MISSINGDATA)
		errors = []
		sigmas = np.linspace(0, 4, 40)
		for sigma in sigmas:
			noise['sigma'] = sigma

			for_evaluation = generate_data(config, 4, '_smacof_with_anchors_single', filter_noise=False, add_noise=noise)

			# generate rmse from coordinates, remembering to not pass last_n_coords to function
			error = [evaluation.rmse(*coords[:2]) for coords in islice(for_evaluation, None, 100)]
			errors.append(error)
		errors = np.array(errors)
		first_q, median, third_q = evaluation.first_third_quartile_and_median(errors)
		#np.save()
		handles = evaluation.plot_rmse_vs_sigma(first_q, median, third_q, x_axis=sigmas, algorithm=ALGORITHMS[i])
		plot_hanldes.extend(handles)
		plot_labels.extend([ALGORITHMS[i], 'IQR boundaries'])
	plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_hanldes)
	plt.show()



	# ALGORITHMS = ['MDS-A, no missing data', 'modified MDS-A, missing inter-tag data']
	# noise = {'mu': 0, 'sigma': 3}
	# plot_hanldes = []
	# plot_labels = []
	# for i in range(2):
	# 	if i == 0:
	# 		config.MISSINGDATA = None
	# 	else:
	# 		config.MISSINGDATA = 'yes'
	# 	print(config.MISSINGDATA)
	# 	errors = []
	# 	no_of_tags = range(1, 30)
	# 	for nt in no_of_tags:
	# 		config.NO_OF_TAGS = nt
	# 		for_evaluation = generate_data(config, 4, '_smacof_with_anchors_single', filter_noise=False, add_noise=noise)

	# 		# generate rmse from coordinates, remembering to not pass last_n_coords to function
	# 		error = [evaluation.rmse(*coords[:2]) for coords in islice(for_evaluation, None, 100)]
	# 		errors.append(error)
	# 	errors = np.array(errors)
	# 	first_q, median, third_q = evaluation.first_third_quartile_and_median(errors)
	# 	#np.save()
	# 	handles = evaluation.plot_rmse_vs_tags(first_q, median, third_q, x_axis=no_of_tags, algorithm=ALGORITHMS[i])
	# 	plot_hanldes.extend(handles)
	# 	plot_labels.extend([ALGORITHMS[i], 'IQR boundaries'])
	# plt.legend(loc='best', fontsize=12, labels=plot_labels, handles=plot_hanldes)
	# axes = plt.gca()
	# axes.set_ylim([0, 3.1])
	# plt.show()




