import cv2
import numpy as np
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
	theta_dist1 =(p1[1]-p2[1]) ** 2
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

	lines = cv2.HoughLines(canny, 1, np.pi/180, 100)

	data = lines[:, 0].copy()

	data[:, 0] = data[:, 0] / np.max(np.abs(data[:, 0]))
	data[:, 1] = data[:, 1] / np.pi

	indices, clusters = dbscan(data, 0.01, min_samples=4, metric=rho_theta_distance)

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

	best_lines = [list(lines[i][0]) for i in firsts]

	return best_lines


def find_board_edges(img):
	lines = find_lines(img)

	hor = np.array([line for line in lines if np.pi / 4 < line[1] < 3 * np.pi / 4])
	ver = np.array([line for line in lines if line not in hor])

	return hor[np.argmin(np.abs(hor[:, 0]))], ver[np.argmin(np.abs(ver[:, 0]))], \
		hor[np.argmax(np.abs(hor[:, 0]))], ver[np.argmax(np.abs(ver[:, 0]))]


def find_chessboard(img):

	edges = find_board_edges(img)

	return edges


