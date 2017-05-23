import numpy as np
from matplotlib import pyplot as plt


COLORS = iter(['blue', 'red', 'green', 'magenta'])

def rmse(computed, real):
	return np.sqrt(((computed - real)**2).mean())


def first_third_quartile_and_median(data):
	first_quartile = np.percentile(data, 25, axis=1)
	third_quartile = np.percentile(data, 75, axis=1)
	median = np.median(data, axis=1)
	return first_quartile, median, third_quartile


def plot_rmse(first_quartile, median, third_quartile, x_axis=None):
	plt.figure(1)
	if x_axis is None:
		x_axis = np.arange(median.shape[0])
	color = next(COLORS)
	plot1, = plt.plot(x_axis, median, 'k-', color=color)
	plot2 = plt.fill_between(x_axis, first_quartile, third_quartile, color=color, alpha=0.3)
	axes = plt.gca()
	return plot1, plot2, axes

def plot_rmse_vs_noise(*args, **kwargs):
	plot1, plot2, axes = plot_rmse(*args, **kwargs)
	axes.set_xlabel('Measurement noise $\sigma$ [m]')
	axes.set_ylabel('RMSE of computed configuration [m]')
	return plot1, plot2

def plot_rmse_vs_anchors(*args, **kwargs):
	plot1, plot2, axes = plot_rmse(*args, **kwargs)
	axes.set_xlabel('No of anchors')
	axes.set_ylabel('RMSE of computed configuration [m]')
	return plot1, plot2

def plot_rmse_vs_ntags(*args, **kwargs):
	plot1, plot2, axes = plot_rmse(*args, **kwargs)
	axes.set_xlabel('No of tags')
	axes.set_ylabel('RMSE of computed configuration [m]')
	return plot1, plot2

	