

if __name__ == '__main__':
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