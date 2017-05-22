import numpy as np


# TODO: use component wise velocities
def compute_velocity(pt1, pt2, time):
	dx, dy = pt2 - pt1
	return dx/time, dy/time