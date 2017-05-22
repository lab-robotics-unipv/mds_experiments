from functools import partial
from core.config import Config
from utils import generate_static_nodes
from experiments import evaluation


ALGORITHMS = ['MDS-A, no missing data', 'modified MDS-A, missing inter-tag data'] 
if __name__ == '__main__':
	
	plot_hanldes = []
	plot_labels = []
	config = Config(no_of_anchors=4, no_of_tags=30)
	generate_data = partial(generate_static_nodes, algorithm='_smacof_with_anchors_single')
	for i in range(2):
		if i == 0:
			# use full distance matrix
			config.missingdata = False
		else:
			# use distance matrix with tag-to-tag distances removed
			config.missingdata = True

		print(config.MISSINGDATA)
		errors = []
		sigmas = np.linspace(0, 4, 40)
		for sigma in sigmas:
			config.sigma = sigma

			# generate rmse from coordinates, remembering to not pass last_n_coords to function
			error = [evaluation.rmse(*generate_data(config=config)[:2]) for i in range(100)]
			errors.append(error)
		errors = np.array(errors)
		handles = evaluation.plot_rmse_vs_sigma(first_q, median, third_q, x_axis=sigmas, algorithm=ALGORITHMS[i])
		plot_hanldes.extend(handles)
		plot_labels.extend([ALGORITHMS[i], 'IQR boundaries'])
	plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_hanldes)
	plt.show()