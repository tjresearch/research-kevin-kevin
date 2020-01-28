import cv2
import numpy as np
import utils


def quadrilateral_slice(points, arr):
	x = np.linspace(0, arr.shape[1] - 1, arr.shape[1])
	y = np.linspace(0, arr.shape[0] - 1, arr.shape[0])
	xx, yy = np.meshgrid(x, y)

	lines = [utils.get_line(points[i], points[(i + 1) % 4]) for i in range(4)]
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


def regioned_segment_board(img, corners, SQ_SIZE):
	SQ_SIZE = 100

	dst_size = SQ_SIZE * 8
	dst_points = [(SQ_SIZE, SQ_SIZE), (SQ_SIZE, dst_size - SQ_SIZE), (dst_size - SQ_SIZE, dst_size - SQ_SIZE), (dst_size - SQ_SIZE, SQ_SIZE)]
	H = utils.find_homography(corners, dst_points)

	chunks = []

	for i in range(8):
		for j in range(8):
			raw_corners = ((i * SQ_SIZE, j * SQ_SIZE),
					   (i * SQ_SIZE, (j + 1) * SQ_SIZE),
					   ((i + 1) * SQ_SIZE, (j + 1) * SQ_SIZE),
					   ((i + 1) * SQ_SIZE, j * SQ_SIZE))

			warped_corners = [utils.inverse_warp_point(raw_corners[k], H) for k in range(4)]

			#square tucked in by margins
			#top right bot left
			margin = [int(SQ_SIZE*pct) for pct in (0.15, 0.15, 0.5, 0.15)]
			region_corners = (
				(raw_corners[0][0]+margin[0], raw_corners[0][1]+margin[1]),
				(raw_corners[1][0]+margin[0], raw_corners[1][1]-margin[1]),
				(raw_corners[2][0]-margin[2], raw_corners[2][1]-margin[3]),
				(raw_corners[3][0]-margin[2], raw_corners[3][1]+margin[3])
			)
			# warped_region_corners = [utils.inverse_warp_point(region_corners[k], H) for k in range(4)]

			# print("half",region_corners)
			# print("full",raw_corners)

			center = ((i + 0.5) * SQ_SIZE, (j + 0.5) * SQ_SIZE)
			center = utils.inverse_warp_point(center, H)

			piece = False

			# square = (np.array(warped_corners), center, np.array(warped_region_corners))
			square = (np.array(warped_corners), center, np.array(region_corners))
			chunks.append(square)

	return chunks, H
