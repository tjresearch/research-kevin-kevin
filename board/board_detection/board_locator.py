import cv2
import numpy as np
import itertools
import utils
import math
import line_detection
from sklearn.cluster import dbscan
from matplotlib import pyplot as plt


def line_weight(line, canny):
	m = -np.cos(line[1]) / np.sin(line[1])
	b = line[0] / np.sin(line[1])
	endpoints = [(0, b), (canny.shape[1] - 1, m * (canny.shape[1] - 1) + b),
				 (-b / m, 0), ((canny.shape[0] - 1 - b) / m, canny.shape[0] - 1)]
	endpoints = [p for p in endpoints if 0 <= p[0] < canny.shape[1] and 0 <= p[1] < canny.shape[0]]
	mag = int(np.hypot(endpoints[0][0] - endpoints[1][0], endpoints[0][1] - endpoints[1][1]))
	xlin = np.linspace(endpoints[0][0], endpoints[1][0], mag)
	ylin = np.linspace(endpoints[0][1], endpoints[1][1], mag)
	weight = 0
	for i in range(mag):
		weight += canny[int(ylin[i]), int(xlin[i])] // 255
	return weight


def rho_theta_distance(p1, p2):

	rho_dist1 = (p1[0] - p2[0]) ** 2
	theta_dist1 = (p1[1] - p2[1]) ** 2
	rho_dist2 = (p1[0] + p2[0]) ** 2
	if p1[1] < p2[1]:
		theta_dist2 = (p1[1] - p2[1] + 1) ** 2
	else:
		theta_dist2 = (p2[1] - p1[1] + 1) ** 2
	return min(np.sqrt(rho_dist1 + theta_dist1), np.sqrt(rho_dist2 + theta_dist2))


def find_lines(img): # Will be implemented with more advanced algorithms later

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	canny = cv2.Canny(gray, 60, 120)

	# cv2.imshow("canny", canny)

	lines = cv2.HoughLines(canny, 1, np.pi/180, 90)

	lines = lines[:, 0]

	return filter_lines(lines)


def find_lines_improved(img):
	lines = line_detection.find_lines(img)
	rho_theta_lines = []
	for line in lines:
		rho_theta_lines.append(utils.convert_ab_to_rho_theta(line))
	return rho_theta_lines
	# return filter_lines(np.array(rho_theta_lines))


def filter_lines(lines):
	data = lines.copy()

	data[:, 0] = data[:, 0] / np.max(np.abs(data[:, 0]))
	data[:, 1] = data[:, 1] / np.pi

	indices, clusters = dbscan(data, 0.02, min_samples=1, metric=rho_theta_distance)

	lines = lines[indices]
	clusters = clusters[indices]
	num_clusters = len(set(clusters))

	firsts = [clusters.tolist().index(i) for i in range(num_clusters)]

	# plt.scatter(lines[:, 0, 0], lines[:, 0, 1], c=clusters)
	# title = "number of cluster: {}".format(num_clusters)
	# plt.title(title)
	# plt.xlabel("Rho")
	# plt.ylabel("Theta")
	# plt.show()

	best_lines = [list(lines[i]) for i in firsts]

	return best_lines


def get_intersections(lines):
	intersections = []
	for i in range(len(lines)):
		for j in range(i + 1, len(lines)):
			intersection = utils.find_intersection(lines[i], lines[j])
			if intersection is not None:
				try:
					intersections.append((int(intersection[0]), int(intersection[1])))
				except OverflowError:
					pass
	return intersections


def find_chessboard(img):
	lines = find_lines_improved(img)

	lattice_points = [p for p in get_intersections(lines) if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]]

	corners = []

	best_board = (None, 0, math.inf)

	for edges in itertools.combinations(lines, 4):
		corners = [p for p in get_intersections(edges) if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]]
		if len(corners) == 4:
			disp = img.copy()
			corners = utils.sorted_ccw(corners)

			# for lattice_point in lattice_points:
			# 	cv2.circle(disp, (int(lattice_point[0]), int(lattice_point[1])), 3, (0, 0, 255), 2)
			#
			# for corner in corners:
			# 	cv2.circle(disp, (int(corner[0]), int(corner[1])), 3, (255, 0, 0), 2)

			square_size = 100
			H = utils.find_homography(corners, square_size)

			close_points = 0
			total_dist = 0

			matched = []

			for i in range(9):
				skip = False
				for j in range(9):
					try:
						point = utils.inverse_warp_point((i * square_size, j * square_size), H)
					except np.linalg.LinAlgError:
						skip = True
						break
					# cv2.circle(disp, (int(point[0]), int(point[1])), 3, (0, 255, 0), 2)
					for lattice_point in lattice_points:
						if utils.dist(point, lattice_point) < 20 and lattice_point not in matched:
							matched.append(lattice_point)
							close_points += 1
							total_dist += utils.dist(point, lattice_point)
							break
				if skip:
					break

			if close_points > best_board[1] or (close_points == best_board[1] and total_dist < best_board[2]):
				best_board = (corners, close_points, total_dist)

			# cv2.imshow("corners", disp)
			# cv2.waitKey()

	return best_board[0]

	# hor = np.array([line for line in lines if np.pi / 4 < line[1] < 3 * np.pi / 4])
	# ver = np.array([line for line in lines if line not in hor])
	#
	# return hor[np.argmin(np.abs(hor[:, 0]))], ver[np.argmin(np.abs(ver[:, 0]))], \
	# 	   hor[np.argmax(np.abs(hor[:, 0]))], ver[np.argmax(np.abs(ver[:, 0]))]
