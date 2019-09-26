import numpy as np
import math

def slope_intercept(line):
	"""Takes two points from a line and return the slope intercept form as a tuple. -> (m, b)"""

	p1, p2 = line

	m = (p2[1] - p1[1]) / (p2[0] - p1[0])
	b = p1[1] - m * p1[0]

	return m, b

def find_intersection(l1, l2):
	"""Finds the intersection between two lines."""

	m1, b1 = slope_intercept(l1)
	m2, b2 = slope_intercept(l2)

	A = np.linalg.inv(np.array([[-m1, 1], [-m2, 1]]))
	B = np.array([[b1], [b2]])

	return tuple(np.matmul(A, B))

def dist(p1, p2):
	return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def angle(l1, l2):
	v1 = np.array([l1[1][0] - l1[0][0], l1[1][1] - l1[0][1]])
	v2 = np.array([l2[1][0] - l2[0][0], l2[1][1] - l2[0][1]])
	mag1 = np.sqrt(np.dot(v1))
	mag2 = np.sqrt(np.dot(v2))
	return np.arccos(np.dot(v1, v2) / (mag1 * mag2))

def calc_gamma(l1, l2):
	inter = find_intersection(l1, l2)
	return (dist(l1[0], inter) + dist(l1[1], inter) + dist(l2[0], inter) + dist(l2[1], inter)) * np.sin(angle(l1, l2)) / 4

def linkable(l1, l2, img):
	omega = np.pi / (2 * math.pow(np.dot(img.shape), 1/4))
	p = 0.9
	t = p * omega
	mag1 = np.sqrt((l1[1][0] - l1[0][0]) ** 2 + (l1[1][1] - l1[0][1]) ** 2)
	mag2 = np.sqrt((l2[1][0] - l2[0][0]) ** 2 + (l2[1][1] - l2[0][1]) ** 2)
	delta = (mag1 + mag2) * t
	gamma = calc_gamma(l1, l2)
	return (mag1 / gamma > delta) ^ (mag2 / gamma > delta)

