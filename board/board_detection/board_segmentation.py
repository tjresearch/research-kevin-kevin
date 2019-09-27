import board_locator
import cv2
import os
import numpy as np


def inverse_warp_point(point, H):
	H = np.linalg.inv(H)
	denom = H[2][0] * point[0] + H[2][1] * point[1] + H[2][2]

	return (H[0][0] * point[0] + H[0][1] * point[1] + H[0][2]) / denom, (
			H[1][0] * point[0] + H[1][1] * point[1] + H[1][2]) / denom


def find_intersection(l1, l2):
	r1, t1 = l1
	r2, t2 = l2

	a = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
	b = np.array([[r1], [r2]])

	try:
		return np.matmul(np.linalg.inv(a), b).reshape(2).tolist()
	except np.linalg.LinAlgError:
		return None


def get_intersections(lines):
	intersections = []
	for i in range(len(lines)):
		for j in range(i + 1, len(lines)):
			intersection = find_intersection(lines[i], lines[j])
			if intersection is not None:
				try:
					intersections.append((int(intersection[0]), int(intersection[1])))
				except OverflowError:
					pass
	return intersections


def get_line(p1, p2):
	if p1[0] == p2[0]:
		return None, p1[0]

	m = (p2[1] - p1[1]) / (p2[0] - p1[0])
	b = p1[1] - m * p1[0]

	return m, b


def quadrilateral_slice(points, arr):
	x = np.linspace(0, arr.shape[1] - 1, arr.shape[1])
	y = np.linspace(0, arr.shape[0] - 1, arr.shape[0])
	xx, yy = np.meshgrid(x, y)

	lines = [get_line(points[i], points[(i + 1) % 4]) for i in range(4)]
	if lines[0][0] is None:
		eq0 = "xx > lines[0][1]"
	elif lines[0][0] > 0:
		eq0 = "yy < lines[0][0] * xx + lines[0][1]"
	else:
		eq0 = "yy > lines[0][0] * xx + lines[0][1]"
	if lines[2][0] is None:
		eq2 = "xx < lines[2][1]"
	elif lines[2][0] > 0:
		eq2 = "yy > lines[2][0] * xx + lines[2][1]"
	else:
		eq2 = "yy < lines[2][0] * xx + lines[2][1]"
	indices = np.all([eval(eq0), yy < lines[1][0] * xx + lines[1][1], eval(eq2), yy > lines[3][0] * xx + lines[3][1]], axis=0)
	return arr * np.repeat(indices[..., np.newaxis], 3, axis=2)


def segment_board(img, edges):

	intersections = []

	for i in range(4):
		intersection = find_intersection(edges[i], edges[(i + 1) % 4])
		intersections.append((int(intersection[0]), int(intersection[1])))

	square_size = 100
	dst_size = 8 * square_size
	dst_corners = [(0, 0), (0, dst_size), (dst_size, dst_size), (dst_size, 0)]

	H, _ = cv2.findHomography(np.array(intersections), np.array(dst_corners))

	chunks = []

	for i in range(8):
		for j in range(8):
			square = []

			corners = [(i * square_size, j * square_size),
					   (i * square_size, (j + 1) * square_size),
					   ((i + 1) * square_size, (j + 1) * square_size),
					   ((i + 1) * square_size, j * square_size)]
			for k in range(4):
				corners[k] = inverse_warp_point(corners[k], H)

			square.append(np.array(corners))

			center = ((i + 0.5) * square_size, (j + 0.5) * square_size)

			center = inverse_warp_point(center, H)

			square.append(center)

			chunks.append(square)

	return chunks

filename = "chessboard4.jpg"
img = cv2.imread(os.path.join("images", filename))

lines = board_locator.find_lines(img)
line_disp = img.copy()

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
	cv2.line(line_disp, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow("lines", line_disp)

edges = board_locator.find_chessboard(img)
chunks = segment_board(img, edges)

chunk_disp = img.copy()

for chunk in chunks:
	corners, center = chunk
	top = np.min(corners[:, 1])
	bottom = np.max(corners[:, 1])
	left = np.min(corners[:, 0])
	right = np.max(corners[:, 0])
	box = ((left, bottom), (right, bottom), (right, bottom - 150), (left, bottom - 150))
	for i in range(4):
		cv2.line(chunk_disp, (int(box[i][0]), int(box[i][1])), (int(box[(i+1)%4][0]), int(box[(i+1)%4][1])), (255, 0, 0), 2)

cv2.imshow("chunks", chunk_disp)

cv2.waitKey()