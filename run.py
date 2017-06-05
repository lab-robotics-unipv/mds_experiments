import argparse

from core.config import Config
from experiments import missing_data, comparisons


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_no', type=int, required=True, help='specify experiment number: 1-4')
	parser.add_argument('--nruns', type=int, required=True, help='number of runs for experiment')
	args = parser.parse_args()
	exp_no, nruns = args.exp_no, args.nruns

	config = Config(no_of_anchors=4, no_of_tags=30, noise=2)
	if exp_no == 1:
		config.missingdata = True
		comparisons.rmse_vs_noise(config=config, no_of_trials=nruns)
	elif exp_no == 2:
		config.missingdata = True
		comparisons.rmse_vs_anchors(config=config, no_of_trials=nruns)
	elif exp_no == 3:
		missing_data.rmse_vs_noise(config=config, no_of_trials=nruns)
	elif exp_no == 4:
		missing_data.rmse_vs_ntags(config=config, no_of_trials=nruns)