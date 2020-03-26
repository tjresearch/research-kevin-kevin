import numpy as np
import utils

def angle(l1, l2):
	v1 = np.array([l1[1][0] - l1[0][0], l1[1][1] - l1[0][1]])
	v2 = np.array([l2[1][0] - l2[0][0], l2[1][1] - l2[0][1]])
	mag1 = np.sqrt(np.dot(v1, v1))
	mag2 = np.sqrt(np.dot(v2, v2))
	return np.arccos(np.dot(v1, v2) / (mag1 * mag2))

def calc_gamma(l1, l2):
	return (utils.line_point_dist(l1, l2[0]) + utils.line_point_dist(l1, l2[1]) +
			utils.line_point_dist(l2, l1[0]) + utils.line_point_dist(l2, l1[1])) / 4 + 0.00001

def linkable(l1, l2, img, t):
	mag1 = np.sqrt((l1[1][0] - l1[0][0]) ** 2 + (l1[1][1] - l1[0][1]) ** 2)
	mag2 = np.sqrt((l2[1][0] - l2[0][0]) ** 2 + (l2[1][1] - l2[0][1]) ** 2)
	delta = (mag1 + mag2) * t
	gamma = calc_gamma(l1, l2)
	return (mag1 / gamma > delta) and (mag2 / gamma > delta)

def link(lines):
	points = []
	for line in lines:
		mag = int(np.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1]))
		x = np.linspace(line[0][0], line[1][0], mag)
		y = np.linspace(line[0][1], line[1][1], mag)
		for i in range(mag):
			points.append((x[i], y[i]))
	points = np.array(points)
	return utils.line_of_best_fit(points)
