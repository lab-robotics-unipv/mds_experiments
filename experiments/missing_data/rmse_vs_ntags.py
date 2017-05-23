from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from core.config import Config
from utils import generate_static_nodes
from experiments import evaluation


ALGORITHMS = ['MDS-A, no missing data', 'modified MDS-A, missing inter-tag data'] 


def runexperiment(no_of_trials=10):
	plot_hanldes = []
	plot_labels = []
	config = Config(no_of_anchors=4, noise=2)
	generate_data = partial(generate_static_nodes, algorithm='_smacof_with_anchors_single')
	for i in range(2):
		if i == 0:
			# use full distance matrix
			config.missingdata = False
		else:
			# use distance matrix with tag-to-tag distances removed
			config.missingdata = True

		errors = []
		no_of_tags = range(1, 30)
		for nt in no_of_tags:
			config.no_of_tags = nt
			# generate rmse from coordinates, remembering to not pass last_n_coords to function
			error = [evaluation.rmse(*generate_data(config=config, add_noise=True)[:2]) for i in range(no_of_trials)]
			errors.append(error)
		errors = np.array(errors)
		first_q, median, third_q = evaluation.first_third_quartile_and_median(errors)
		handles = evaluation.plot_rmse_vs_tags(first_q, median, third_q, x_axis=no_of_tags, algorithm=ALGORITHMS[i])
		plot_hanldes.extend(handles)
		plot_labels.extend([ALGORITHMS[i], 'IQR boundaries'])
	plt.legend(loc='best', fontsize=12, labels=plot_labels, handles=plot_hanldes)
	axes = plt.gca()
	axes.set_ylim([0, 3.1])
	plt.show()