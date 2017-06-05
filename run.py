from core.config import Config
from experiments import missing_data, comparisons


if __name__ == '__main__':
	# config = Config(no_of_anchors=4, no_of_tags=30, noise=2, missingdata=True)
	# comparisons.rmse_vs_noise(config=config, no_of_trials=5)

	# comparisons.rmse_vs_anchors(config=config, no_of_trials=5)

	config = Config(no_of_anchors=4, no_of_tags=30, noise=2)
	#missing_data.rmse_vs_noise(config=config, no_of_trials=5)

	missing_data.rmse_vs_ntags(config=config, no_of_trials=5)