from matplotlib import pyplot as plt
from matplotlib import animation

from core.config import Config
from utils import generate_dynamic_nodes

# number of mobile node transitions
NO_TRANS = 100

class NodeAnimation(object):
	'''create animation of tag-anchor deployment from a given configuration of parameters'''

	def __init__(self, cfg, data, show_trail=False):
		self.fig = plt.figure()
		self.data = data
		self.show_trail = show_trail
		self.no_of_anchors = cfg.no_of_anchors
		self.no_of_tags = cfg.no_of_tags

	def init_plot(self):
		real_coords, mds_coords, *others = next(self.data)
		self.anchors_scat = plt.scatter(real_coords[:self.no_of_anchors, 0], real_coords[:self.no_of_anchors, 1], color='blue', s=100, lw=1, label='Anchors positions', marker='o')
		self.real_scat = plt.scatter(real_coords[self.no_of_anchors:, 0], real_coords[self.no_of_anchors:, 1], color='blue', s=100, lw=1, label='Tag positions', marker='o', facecolors='none')
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
	cfg = Config(no_of_anchors=4, no_of_tags=7, missingdata=True, sigma=0)
	data = generate_dynamic_nodes(cfg, algorithm='_smacof_with_anchors_single', no_of_trans=NO_TRANS, add_noise=True, filter_noise=False)
	anim = NodeAnimation(cfg, data, show_trail=False)
	anim.draw_plot()
