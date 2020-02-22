import numpy as np
import itertools
import pyclipper
import matplotlib.path
import utils
import math
import cv2
import sklearn
import scipy.spatial, scipy.cluster
import collections
import line_detection
import tensorflow as tf

def load_model(model_file, weights_file):
	json_file = open(model_file, "r")
	model_json = json_file.read()
	json_file.close()
	model = tf.keras.models.model_from_json(model_json)
	model.load_weights(weights_file)
	return model

def validate_lattice_points(model, lattice_points, img):

	poss_points = []
	poss_images = []
	for lattice_point in lattice_points:
		if 10 < lattice_point[0] < img.shape[1] - 10 and 10 < lattice_point[1] < img.shape[0] - 10:
			subimg = img[lattice_point[1] - 10:lattice_point[1] + 11, lattice_point[0] - 10:lattice_point[0] + 11]

			subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
			subimg = cv2.threshold(subimg, 0, 255, cv2.THRESH_OTSU)[1]
			subimg = cv2.Canny(subimg, 0, 255)

			subimg = subimg.astype(np.float32) / 255.0

			poss_points.append(lattice_point)
			poss_images.append(subimg)

	poss_images = np.array(poss_images)
	conf = np.argmax(model.predict(poss_images[..., np.newaxis]), axis=1)

	return poss_points, conf


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


def cluster_points(points, max_dist=10):
	Y = scipy.spatial.distance.pdist(points)
	Z = scipy.cluster.hierarchy.single(Y)
	T = scipy.cluster.hierarchy.fcluster(Z, max_dist, "distance")
	clusters = collections.defaultdict(list)

	for i in range(len(T)):
		clusters[T[i]].append(points[i])

	clusters = clusters.values()
	clusters = map(lambda arr : (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), clusters)

	return list(clusters)


def find_lattice_points(img, lines, lattice_point_model):

	intersections = [p for p in get_intersections(lines) if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]]

	# lattice_disp = img.copy()
	#
	# for line in lines:
	# 	rho, theta = line
	# 	a = np.cos(theta)
	# 	b = np.sin(theta)
	# 	x0 = a * rho
	# 	y0 = b * rho
	# 	x1 = int(x0 + 1000 * -b)
	# 	y1 = int(y0 + 1000 * a)
	# 	x2 = int(x0 - 1000 * -b)
	# 	y2 = int(y0 - 1000 * a)
	# 	cv2.line(lattice_disp, (x1, y1), (x2, y2), (255, 0, 0), 4)

	poss_points, conf = validate_lattice_points(lattice_point_model, intersections, img)
	lattice_points = np.array(poss_points)[np.nonzero(conf)]

	lattice_points = cluster_points(lattice_points)

	# for point in lattice_points:
	# 	cv2.circle(lattice_disp, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
	#
	# cv2.imshow("lattice", lattice_disp)
	# cv2.waitKey()

	return lattice_points


def polyscore(corners, lattice_points, centroid, alpha, beta):
	corners = utils.sorted_ccw(corners)
	area = cv2.contourArea(np.array(corners))
	if area < 20 * alpha ** 2:
		return 0

	gamma = alpha / 1.5

	offset = pyclipper.PyclipperOffset()
	offset.AddPath(corners, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
	pcorners = matplotlib.path.Path(offset.Execute(gamma)[0])
	bool_points_inside = pcorners.contains_points(lattice_points)

	num_points_inside = min(np.count_nonzero(bool_points_inside), 49)
	if num_points_inside < min(len(lattice_points), 49) - 2 * beta - 1:
		return 0

	points_inside = []
	for i in range(len(bool_points_inside)):
		if bool_points_inside[i]:
			points_inside.append(lattice_points[i])

	hull = scipy.spatial.ConvexHull(np.array(points_inside)).vertices
	hull = np.array([lattice_points[p] for p in hull])

	frame_centroid = np.mean(hull, axis=0)
	centroid_dist = np.linalg.norm(np.array(centroid) - frame_centroid)

	lines = [[corners[0], corners[1]], [corners[1], corners[2]], [corners[2], corners[3]], [corners[3], corners[0]]]
	avg_dist = 0
	num_points = 0
	for line in lines:
		for point in hull:
			dist = utils.line_point_dist(line, point)
			if dist < gamma:
				avg_dist += dist
				num_points += 1

	if num_points == 0:
		return 0

	avg_dist /= num_points

	W = lambda i, x : 1 / (1 + math.pow(x / area, 1 / i))

	return math.pow(num_points_inside, 4) / math.pow(area, 2) * W(3, avg_dist) * W(5, centroid_dist)


def find_chessboard(img, lattice_point_model):
	lines = line_detection.find_lines_rho_theta(img)

	lattice_points = find_lattice_points(img, lines, lattice_point_model)

	lattice_points = utils.sorted_ccw(lattice_points)

	for i in range(len(lattice_points)):
		lattice_points[i] = (int(lattice_points[i][0]), int(lattice_points[i][1]))

	alpha = math.sqrt(cv2.contourArea(np.array(lattice_points)) / 49)
	beta = len(lattice_points) / 20

	X = sklearn.cluster.DBSCAN(eps=alpha * 4).fit(lattice_points)

	point_clusters = {}
	for i in range(len(lattice_points)):
		if X.labels_[i] != -1:
			if X.labels_[i] in point_clusters:
				point_clusters[X.labels_[i]].append(lattice_points[i])
			else:
				point_clusters[X.labels_[i]] = [lattice_points[i]]

	points = []
	for i in point_clusters:
		if len(point_clusters[i]) > len(points):
			points = point_clusters[i]

	centroid = list(np.mean(np.array(points), axis=0))
	alpha = math.sqrt(cv2.contourArea(np.array(points)) / 49)

	vertical = []
	horizontal = []
	for line in lines:
		for point in points:
			t1 = utils.rho_theta_line_point_dist(line, point) < alpha
			t2 = utils.rho_theta_line_point_dist(line, centroid) > 2.5 * alpha
			if t1 and t2:
				if np.pi / 4 < line[1] < 3 * np.pi / 4:
					if line not in horizontal:
						horizontal.append(line)
				else:
					if line not in vertical:
						vertical.append(line)

	best_board = []
	best_polyscore = 0
	for v in itertools.combinations(vertical, 2):
		for h in itertools.combinations(horizontal, 2):
			board_lines = [v[0], v[1], h[0], h[1]]
			corners = get_intersections(board_lines)
			corners = [corner for corner in corners if 0 <= corner[0] < img.shape[1] and 0 <= corner[1] < img.shape[0]]
			if len(corners) == 4:
				p = polyscore(corners, lattice_points, centroid, alpha / 2, beta)
				# print(p, corners)
				if p > best_polyscore:
					best_board = [board_lines, corners]
					best_polyscore = p

	# print(best_board)

	# line_disp = img.copy()
	# for line in best_board[0]:
	# 	rho, theta = line
	# 	a = np.cos(theta)
	# 	b = np.sin(theta)
	# 	x0 = a * rho
	# 	y0 = b * rho
	# 	x1 = int(x0 + 1000 * -b)
	# 	y1 = int(y0 + 1000 * a)
	# 	x2 = int(x0 - 1000 * -b)
	# 	y2 = int(y0 - 1000 * a)
	# 	cv2.line(line_disp, (x1, y1), (x2, y2), (255, 0, 0), 4)
	#
	# cv2.imshow("lines", line_disp)
	# cv2.waitKey()

	best_board[1] = utils.sorted_ccw(best_board[1])

	return best_board