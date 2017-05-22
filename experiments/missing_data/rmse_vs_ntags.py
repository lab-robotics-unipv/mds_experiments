

if __name__ == '__main__':
	ALGORITHMS = ['MDS-A, no missing data', 'modified MDS-A, missing inter-tag data']
	noise = {'mu': 0, 'sigma': 3}
	plot_hanldes = []
	plot_labels = []
	for i in range(2):
		if i == 0:
			config.MISSINGDATA = None
		else:
			config.MISSINGDATA = 'yes'
		print(config.MISSINGDATA)
		errors = []
		no_of_tags = range(1, 30)
		for nt in no_of_tags:
			config.NO_OF_TAGS = nt
			for_evaluation = generate_data(config, 4, '_smacof_with_anchors_single', filter_noise=False, add_noise=noise)

			# generate rmse from coordinates, remembering to not pass last_n_coords to function
			error = [evaluation.rmse(*coords[:2]) for coords in islice(for_evaluation, None, 100)]
			errors.append(error)
		errors = np.array(errors)
		first_q, median, third_q = evaluation.first_third_quartile_and_median(errors)
		#np.save()
		handles = evaluation.plot_rmse_vs_tags(first_q, median, third_q, x_axis=no_of_tags, algorithm=ALGORITHMS[i])
		plot_hanldes.extend(handles)
		plot_labels.extend([ALGORITHMS[i], 'IQR boundaries'])
	plt.legend(loc='best', fontsize=12, labels=plot_labels, handles=plot_hanldes)
	axes = plt.gca()
	axes.set_ylim([0, 3.1])
	plt.show()