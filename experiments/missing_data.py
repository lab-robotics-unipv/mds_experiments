from functools import partial, wraps

import numpy as np
from matplotlib import pyplot as plt

from core.config import Config
from utils import generate_static_nodes
from experiments import evaluation


ALGORITHMS = ['MDS-A, no missing data', 'modified MDS-A, missing inter-tag data'] 


def runexperiment(func):
	@wraps(func)
	def wrapper(**kwargs):
		plot_hanldes = []
		plot_labels = []
		config = Config(no_of_anchors=4, no_of_tags=30, noise=2)
		generate_data = partial(generate_static_nodes, algorithm='_smacof_with_anchors_single', add_noise=True)
		for i in range(2):
			if i == 0:
				# use full distance matrix
				config.missingdata = False
			else:
				# use distance matrix with tag-to-tag distances removed
				config.missingdata = True
			kwargs['config'] =  config
			kwargs['data_func'] = generate_data
			x_axis, errors = func(**kwargs)
			errors = np.array(errors)
			first_q, median, third_q = evaluation.first_third_quartile_and_median(errors)
			handles = getattr(evaluation, 'plot_{}'.format(func.__name__))(first_q, median, third_q, x_axis=x_axis)
			plot_hanldes.extend(handles)
			plot_labels.extend([ALGORITHMS[i], 'IQR boundaries'])
		plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_hanldes)
		plt.show()
	return wrapper

@runexperiment
def rmse_vs_noise(config=None, data_func=None, no_of_trials=10):
	errors = []
	sigmas = np.linspace(0, 4, 40)
	for sigma in sigmas:
		config.sigma = sigma
		# generate rmse from coordinates, remembering to not pass last_n_coords to function
		error = [evaluation.rmse(*data_func(config=config)[:2]) for i in range(no_of_trials)]
		errors.append(error)
	return sigmas, errors

@runexperiment
def rmse_vs_ntags(config=None, data_func=None, no_of_trials=10):
	errors = []
	tag_count = range(1, 30)
	for nt in tag_count:
		config.no_of_tags = nt
		# generate rmse from coordinates, remembering to not pass last_n_coords to function
		error = [evaluation.rmse(*data_func(config=config)[:2]) for i in range(no_of_trials)]
		errors.append(error)
	errors = np.array(errors)
	return tag_count, errors