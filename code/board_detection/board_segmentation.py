# Handles the segmentation of the board into 64 separate squares

import cv2
import os
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

"""
segments board into squares and orthophoto regions given inside corners
called by split_chessboard() in piece_detection/square_splitter.py
"""
def regioned_segment_board(img, corners, SQ_SIZE, graphics_IO=None):
	dst_size = SQ_SIZE * 8
	#ccw from top-left, xy
	dst_points = [(SQ_SIZE, SQ_SIZE), (SQ_SIZE, dst_size - SQ_SIZE), (dst_size - SQ_SIZE, dst_size - SQ_SIZE), (dst_size - SQ_SIZE, SQ_SIZE)]
	H = utils.find_homography(corners, dst_points)
	H_inv = np.linalg.inv(H)

	sqr_corners = []
	top_ortho_regions = []
	for r in range(8):
		for c in range(8):
			#ccw from top-left, xy
			raw_corners = np.float32([[c * SQ_SIZE, r * SQ_SIZE],
					   [c * SQ_SIZE, (r + 1) * SQ_SIZE],
					   [(c + 1) * SQ_SIZE, (r + 1) * SQ_SIZE],
					   [(c + 1) * SQ_SIZE, r * SQ_SIZE]])

			warped_corners = cv2.perspectiveTransform(np.expand_dims(raw_corners, axis=1), H_inv)
			warped_corners = warped_corners[:, 0]

			#square tucked in by margins
			#top left bot right, xy
			margin = [int(SQ_SIZE*pct) for pct in (0.15, 0.15, 0.50, 0.15)]
			region_corners = (
				(raw_corners[0, 0]+margin[1], raw_corners[0, 1]+margin[0]),
				(raw_corners[1, 0]+margin[1], raw_corners[1, 1]-margin[2]),
				(raw_corners[2, 0]-margin[3], raw_corners[2, 1]-margin[2]),
				(raw_corners[3, 0]-margin[3], raw_corners[3, 1]+margin[0])
			)

			sqr_corners.append(warped_corners)
			top_ortho_regions.append(np.array(region_corners).astype(int))

	if graphics_IO:
		out_dir = graphics_IO[1]
		segment_disp = img.copy()

		for square in sqr_corners:
			int_square = np.int0(square)
			for i in range(4):
				cv2.line(segment_disp, tuple(int_square[i]), tuple(int_square[(i + 1) % 4]), (255, 0, 0), 3)

		cv2.imwrite(os.path.join(out_dir, "board_segmentation.jpg"), segment_disp)

	return sqr_corners, top_ortho_regions, H
