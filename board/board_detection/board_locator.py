import cv2
import numpy as np
from sklearn.cluster import dbscan
from matplotlib import pyplot as plt


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

	lines = cv2.HoughLines(canny, 1, np.pi/180, 100)

	data = lines[:, 0].copy()

	data[:, 0] = data[:, 0] / np.max(np.abs(data[:, 0]))
	data[:, 1] = data[:, 1] / np.pi

	indices, clusters = dbscan(data, 0.01, min_samples=3, metric=rho_theta_distance)

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


def find_intersection(l1, l2):
	r1, t1 = l1
	r2, t2 = l2

	a = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
	b = np.array([[r1], [r2]])

	return np.matmul(np.linalg.inv(a), b)


def find_chessboard(img):

	edges = find_board_edges(img)
	intersections = []

	for i in range(4):
		intersection = find_intersection(edges[i], edges[(i+1)%4]).reshape(2).tolist()
		intersections.append((int(intersection[0]), int(intersection[1])))

	margins = 20
	dst_size = 800
	dst_corners = [(margins, margins), (margins, dst_size + margins), (dst_size + margins, dst_size + margins), (dst_size + margins, margins)]

	H, _ = cv2.findHomography(np.array(intersections), np.array(dst_corners))

	warped = cv2.warpPerspective(img, H, (dst_size + margins * 2, dst_size + margins * 2))

	return warped
