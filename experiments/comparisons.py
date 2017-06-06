from functools import partial, wraps

import numpy as np
from matplotlib import pyplot as plt

from core.config import Config, ALGORITHMS
from utils import generate_static_nodes
from experiments import evaluation


LABELS = dict(zip(ALGORITHMS, ['Classical MDS', 'Anchored MDS', 'MDS-RFID']))

def runexperiment(func):
	@wraps(func)
	def wrapper(**kwargs):
		plot_hanldes = []
		plot_labels = []
		for algorithm in ALGORITHMS:
			kwargs['algorithm'] = algorithm
			x_axis, errors = func(**kwargs)
			errors = np.array(errors)
			first_q, median, third_q = evaluation.first_third_quartile_and_median(errors)
			handles = getattr(evaluation, 'plot_{}'.format(func.__name__))(first_q, median, third_q, x_axis=x_axis)
			plot_hanldes.extend(handles)
			plot_labels.extend([LABELS[algorithm], 'IQR boundaries'])
		plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_hanldes)
		plt.show()
	return wrapper

@runexperiment
def rmse_vs_noise(algorithm=None, config=None, no_of_trials=10):
	errors = []
	sigmas = np.linspace(0, 4, 40)
	for sigma in sigmas:
		config.sigma = sigma
		# generate rmse from coordinates, remembering to not pass last_n_coords to function
		error = [evaluation.rmse(*generate_static_nodes(config, algorithm)[:2]) for i in range(no_of_trials)]
		errors.append(error)
	return sigmas, errors

@runexperiment
def rmse_vs_anchors(algorithm=None, config=None, no_of_trials=10):
	errors = []
	anchors = range(3, 9)
	for anc in anchors:
		config.no_of_anchors = anc
		# generate rmse from coordinates, remembering to not pass last_n_coords to function
		error = [evaluation.rmse(*generate_static_nodes(config, algorithm)[:2]) for i in range(no_of_trials)]
		errors.append(error)
	return anchors, errors
