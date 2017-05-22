import numpy as np


def recover_tag_distances(config, prox_arr):
	NO_OF_TAGS, NO_OF_ANCHORS = config.NO_OF_TAGS, config.NO_OF_ANCHORS
	for j in range(NO_OF_ANCHORS, NO_OF_TAGS+NO_OF_ANCHORS):
		for i in range(j, NO_OF_TAGS+NO_OF_ANCHORS):
			if i == j:
				continue
			prox_arr[i, j] = prox_arr[j, i] = np.mean(np.absolute([prox_arr[i,a]-prox_arr[j,a] for a in range(NO_OF_ANCHORS)]))
