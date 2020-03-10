import cv2
import os
import math
import numpy as np
import line_linking
import utils

CLAHE_PARAMS = [[3, (2, 6), 5],
				[3, (6, 2), 5],
				[5, (3, 3), 5],
				[0, (0, 0), 0]]


def canny(img, sigma=0.25):
	"""Applies Canny edge detection to the given image."""

	v = np.median(img)

	img = cv2.medianBlur(img, 5)
	img = cv2.GaussianBlur(img, (7, 7), 2)

	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))

	return cv2.Canny(img, lower, upper)


def line_detector(img):
	"""Applies probabilistic Hough transform to the edge gradient."""

	out = []
	lines = cv2.HoughLinesP(img, 1, np.pi/180, 40, minLineLength=50, maxLineGap=15)

	if lines is None:
		return []

	for line in np.reshape(lines, (-1, 4)):
		out += [[[int(line[0]), int(line[1])], [int(line[2]), int(line[3])]]]

	return out


def clahe(img, limit, grid, iters): # Taken from Czyzewski, et al.
	"""Applies CLAHE with 3 sets of hyperparameters."""

	for i in range(iters):
		img = cv2.createCLAHE(limit, grid).apply(img)

	if limit != 0:
		kernel = np.ones((10, 10), dtype=np.uint8)
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	return img


def find(index, lines):

	while lines[index][1] != index:
		index, lines[index][1] = lines[index][1], lines[lines[index][1]][1]

	return index


def union(index1, index2, lines):

	root1 = find(index1, lines)
	root2 = find(index2, lines)

	if root1 == root2:
		return

	if lines[root1][2] < lines[root2][2]:
		root1, root2 = root2, root1

	lines[root2][1] = root1
	if lines[root1][2] == lines[root2][2]:
		lines[root1][2] += 1


def disp_lines_ab(lines, img):
	disp = img.copy()
	for line in lines:
		a, b = line
		if a == 0:
			endpoints = [(0, 1 / b), (img.shape[1], (1 - img.shape[1] * a) / b)]
		elif b == 0:
			endpoints = [(1 / a, 0), ((1 - img.shape[0] * b) / a, img.shape[0])]
		else:
			intercepts = [(1 / a, 0),
						  ((1 - img.shape[0] * b) / a, img.shape[0]),
						  (0, 1 / b),
						  (img.shape[1], (1 - img.shape[1] * a) / b)]
			endpoints = []
			for intercept in intercepts:
				if 0 <= intercept[0] <= img.shape[1] and 0 <= intercept[1] <= img.shape[0]:
					endpoints.append(intercept)
		cv2.line(disp, (int(endpoints[0][0]), int(endpoints[0][1])),
				 (int(endpoints[1][0]), int(endpoints[1][1])), (255, 0, 0), 2)
	return disp


def find_lines(img, out_dir=None):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	i = 0
	lines = []

	for arr in CLAHE_PARAMS:
		temp = clahe(gray, limit=arr[0], grid=arr[1], iters=arr[2])
		new_lines = line_detector(canny(temp))
		for line in new_lines:
			if line not in lines:
				lines.append(line)

	if out_dir:
		line_disp = img.copy()

		for line in lines:
			cv2.line(line_disp, tuple(line[0]), tuple(line[1]), (255, 0, 0), 2)

		cv2.imwrite(os.path.join(out_dir, "line_detection.jpg"), line_disp)

	for i in range(len(lines)):
		lines[i] = [lines[i], i, 0]  # Line, parent, rank

	omega = np.pi / (2 * math.pow(math.pow(min(img.shape[0], img.shape[1]), 2), 1 / 4))
	p = 0.9
	t = p * omega

	for i in range(len(lines)):
		for j in range(i + 1, len(lines)):
			if line_linking.linkable(lines[i][0], lines[j][0], img, t):
				union(i, j, lines)

	groups = {}

	for i in range(len(lines)):
		if lines[i][1] in groups:
			groups[lines[i][1]].append(lines[i][0])
		else:
			groups[lines[i][1]] = [lines[i][0]]

	linked_lines = []
	for group in groups.values():
		linked_lines.append(line_linking.link(group))

	linked_disp = disp_lines_ab(linked_lines, img)

	if out_dir:
		cv2.imwrite(os.path.join(out_dir, "line_linking.jpg"), linked_disp)

	return linked_lines



def find_lines_rho_theta(img, out_dir=None):
	lines = find_lines(img, out_dir)
	rho_theta_lines = []
	for line in lines:
		rho_theta_lines.append(utils.convert_ab_to_rho_theta(line))
	return rho_theta_lines
