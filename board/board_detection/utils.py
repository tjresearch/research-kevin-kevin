import numpy as np
import math
import cv2


def inverse_warp_point(point, H):
	H = np.linalg.inv(H)
	denom = H[2][0] * point[0] + H[2][1] * point[1] + H[2][2]

	return (H[0][0] * point[0] + H[0][1] * point[1] + H[0][2]) / denom, (
			H[1][0] * point[0] + H[1][1] * point[1] + H[1][2]) / denom


def get_line(p1, p2):
	if p1[0] == p2[0]:
		return None, p1[0]

	m = (p2[1] - p1[1]) / (p2[0] - p1[0])
	b = p1[1] - m * p1[0]

	return m, b


def find_intersection(l1, l2):
	r1, t1 = l1
	r2, t2 = l2

	a = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
	b = np.array([[r1], [r2]])

	try:
		return np.matmul(np.linalg.inv(a), b).reshape(2).tolist()
	except np.linalg.LinAlgError:
		return None

def get_top_left(points):

	min_dist_squared = math.inf
	min_point = None

	for point in points:

		dist_squared = point[0] ** 2 + point[1] ** 2
		if dist_squared < min_dist_squared or dist_squared == min_dist_squared and point[1] > min_point[1]:
			min_point = point
			min_dist_squared = dist_squared

	return min_point


def angle_and_dist(point, origin, refvec):

	vector = [point[0] - origin[0], point[1] - origin[1]]

	lenvector = math.hypot(vector[0], vector[1])
	if lenvector == 0:
		return -math.pi, 0

	normalized = [vector[0] / lenvector, vector[1] / lenvector]

	dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]
	diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]

	angle = math.atan2(diffprod, dotprod)

	if angle < 0:
		return 2 * math.pi + angle, lenvector

	return angle, lenvector


def sorted_ccw(points):
	origin = get_top_left(points)
	refvec = [0, -1]
	return sorted(points, key=lambda x: angle_and_dist(x, origin, refvec))


def find_homography(corners, square_size):
	dst_size = 8 * square_size
	dst_corners = [(0, 0), (0, dst_size), (dst_size, dst_size), (dst_size, 0)]

	H, _ = cv2.findHomography(np.array(corners), np.array(dst_corners))

	return H


def dist(p1, p2):
	return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def line_of_best_fit(points):
	"""Returns line in (a, b) form, where ax + by = 1"""
	xmean, ymean = np.mean(points, axis=0)
	try:
		m = np.sum((points[:, 0] - xmean) * (points[:, 1] - ymean)) / np.sum((points[:, 0] - xmean) ** 2)
	except ZeroDivisionError:
		m = None
	if m is not None and not np.isnan(m):
		yint = ymean - m * xmean
		a = -m / yint
		b = 1 / yint
	else:
		a = 1 / xmean
		b = 0
	return a, b