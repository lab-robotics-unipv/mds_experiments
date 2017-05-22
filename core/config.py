import numpy as np

# number of dimensions of configuration 
M = 2 

# number of mobile node transitions
NO_TRANS = 100

# channel estimates for each anchor as seen by the tag
ALPHA = np.array([1.6, 2.4, 1.9, 1.51, 1.6, 2.0])
ALPHA = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

# define bounding box for node deployment
'''Larger values reduce the chances that nodes will overlap'''
X_MAX = 30
X_MIN = 0
Y_MAX = 20
Y_MIN = 0

ORIGIN = np.array([X_MIN, Y_MIN])
ANCHORS_ = np.array([ORIGIN, 
				   [X_MAX, Y_MIN], 
				   [X_MAX, Y_MAX],
				   [X_MIN, Y_MAX],
				   [(X_MAX+X_MIN)/2.0, Y_MAX],
				   [(X_MAX+X_MIN)/2.0, Y_MIN],
				   [X_MIN, (Y_MAX+Y_MIN)/2.0],
				   [X_MAX, (Y_MAX+Y_MIN)/2.0]])


NO_OF_ANCHORS = len(ANCHORS_[:4])
NO_OF_TAGS = 10

def generate_points(anc, NO_OF_TAGS):
	#TODO: handle overlapping points
	NO_OF_ANCHORS = anc
	if NO_OF_TAGS < 1:
		raise ValueError('number of tags cannot be less that 1')
	x0_displacements = np.random.choice(np.arange(1, X_MAX-X_MIN-1), size=NO_OF_TAGS)
	y0_displacements = np.random.choice(np.arange(1, Y_MAX-Y_MIN-1), size=NO_OF_TAGS)
	tags =  np.array([ORIGIN + pt for pt in zip(x0_displacements, y0_displacements)])
	return np.concatenate((ANCHORS_[:anc], tags))

# POINTS = generate_points(ANCHORS)
DEFAULT_POINTS = np.array([[0,20], [30,20], [30,0], [0,0], [18, 7], [4, 9], [10, 5], [7, 17], [15, 14], [12, 18], [22, 11]])