import numpy as np
import itertools
import utils
import math
import cv2
import line_detection


def separate_lines(lines):
	hor, vert = [], []
	for line in lines:
		if np.pi / 4 < line[1] < 3 * np.pi / 4:
			hor.append(line)
		else:
			vert.append(line)
	return hor, vert


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
	lines = line_detection.find_lines_improved(img)

	lattice_points = [p for p in get_intersections(lines) if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]]

	lattice_disp = img.copy()

	for line in lines:
		rho, theta = line
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a * rho
		y0 = b * rho
		x1 = int(x0 + 1000 * -b)
		y1 = int(y0 + 1000 * a)
		x2 = int(x0 - 1000 * -b)
		y2 = int(y0 - 1000 * a)
		cv2.line(lattice_disp, (x1, y1), (x2, y2), (255, 0, 0), 2)

	for lattice_point in lattice_points:
		cv2.circle(lattice_disp, (int(lattice_point[0]), int(lattice_point[1])), 3, (0, 0, 255), 2)

	cv2.imshow("lattice", lattice_disp)

	corners = []

	best_board = (None, 0, math.inf)

	horizontal, vertical = separate_lines(lines)

	for hor in itertools.combinations(horizontal, 2):
		for vert in itertools.combinations(vertical, 2):
			edges = [*hor, *vert]
			corners = [p for p in get_intersections(edges) if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]]
			if len(corners) == 4:
				disp = img.copy()
				corners = utils.sorted_ccw(corners)

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