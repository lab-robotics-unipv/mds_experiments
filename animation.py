

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