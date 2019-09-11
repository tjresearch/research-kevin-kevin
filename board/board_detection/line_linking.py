import numpy as np

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