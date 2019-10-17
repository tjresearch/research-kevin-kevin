import numpy as np
import math
import utils

def get_line_eq(line):
	p1, p2 = line

	A = np.linalg.inv(np.array([[p1[0], p1[1]], [p2[0], p2[1]]]))
	B = np.array([[1], [1]])

	return np.matmul(A, B)[:, 0].tolist()

def find_intersection(l1, l2):
	a1, b1 = get_line_eq(l1)
	a2, b2 = get_line_eq(l2)

	try:
		A = np.linalg.inv(np.array([[a1, b1], [a2, b2]]))
	except np.linalg.LinAlgError:
		return None

	B = np.array([[1], [1]])

	return np.matmul(A, B)[:, 0].tolist()

def dist(p1, p2): #repeat?
	return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def angle(l1, l2):
	v1 = np.array([l1[1][0] - l1[0][0], l1[1][1] - l1[0][1]])
	v2 = np.array([l2[1][0] - l2[0][0], l2[1][1] - l2[0][1]])
	mag1 = np.sqrt(np.dot(v1, v1))
	mag2 = np.sqrt(np.dot(v2, v2))
	return np.arccos(np.dot(v1, v2) / (mag1 * mag2))

def calc_gamma(l1, l2):
	inter = find_intersection(l1, l2)
	if inter is None:
		return None
	return (dist(l1[0], inter) + dist(l1[1], inter) + dist(l2[0], inter) + dist(l2[1], inter)) * np.sin(angle(l1, l2)) / 4

def linkable(l1, l2, img):
	omega = np.pi / (2 * math.pow(img.shape[0] * img.shape[1], 1/4))
	p = 0.9
	t = p * omega
	mag1 = np.sqrt((l1[1][0] - l1[0][0]) ** 2 + (l1[1][1] - l1[0][1]) ** 2)
	mag2 = np.sqrt((l2[1][0] - l2[0][0]) ** 2 + (l2[1][1] - l2[0][1]) ** 2)
	delta = (mag1 + mag2) * t
	gamma = calc_gamma(l1, l2)
	if gamma is None:
		return False
	return (mag1 / gamma > delta) ^ (mag2 / gamma > delta)


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
