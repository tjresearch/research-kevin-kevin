import numpy as np
import cv2
import os
import board_locator
import board_segmentation
import utils
import time

SQ_SIZE = 100
dst_size = SQ_SIZE * 8
dst_points = [(SQ_SIZE, SQ_SIZE), (SQ_SIZE, dst_size - SQ_SIZE), (dst_size - SQ_SIZE, dst_size - SQ_SIZE), (dst_size - SQ_SIZE, SQ_SIZE)]

def get_color_diff_grid(img1, img2, corners1, corners2):
	H1 = utils.find_homography(corners1, dst_points)
	H2 = utils.find_homography(corners2, dst_points)

	warped1 = cv2.cvtColor(cv2.warpPerspective(img1, H1, (dst_size, dst_size)), cv2.COLOR_BGR2GRAY)
	warped2 = cv2.cvtColor(cv2.warpPerspective(img2, H2, (dst_size, dst_size)), cv2.COLOR_BGR2GRAY)

	split1 = np.concatenate(np.split(np.array(np.split(warped1, 8, axis=0)), 8, axis=2))
	split2 = np.concatenate(np.split(np.array(np.split(warped2, 8, axis=0)), 8, axis=2))

	diff = np.abs(split1.astype(int) - split2.astype(int))
	diff = np.mean(diff, axis=(1, 2))

	diff = np.flip(np.rot90(np.reshape(diff, (8, 8))), 0)

	return diff

def color_diff_display(img, corners, diff_grid):
	sqr_corners, _, _ = board_segmentation.regioned_segment_board(img, corners, 100)
	heatmap = np.zeros(img.shape[:2], np.uint8)
	for i in range(64):
		square = sqr_corners[i]
		int_square = np.expand_dims(np.int0(square), axis=1)
		r = i // 8
		c = i % 8
		cv2.fillConvexPoly(heatmap, int_square, int(diff_grid[r, c] / diff_grid.max() * 255))
	return heatmap

if __name__ == "__main__":
	model_path = "../models"
	lattice_model = board_locator.load_model(os.path.join(model_path, "lattice_points_model.json"), os.path.join(model_path, "lattice_points_model.h5"))

	# img1_path = "./images/imgs_3_15_chung/*game_11/*000.jpeg"
	# img2_path = "./images/imgs_3_15_chung/*game_11/*001.jpeg"
	#
	# img1 = cv2.imread(img1_path)
	# img2 = cv2.imread(img2_path)
	#
	# lines1, corners1 = board_locator.find_chessboard(img1, lattice_model)
	# lines2, corners2 = board_locator.find_chessboard(img2, lattice_model, prev=(img1, corners1))
	#
	# diff_grid = get_color_diff_grid(img1, img2, corners1, corners2)
	#
	# heatmap = color_diff_display(img2, corners2, diff_grid)
	#
	# cv2.imshow("img1", img1)
	# cv2.imshow("img2", img2)
	# cv2.imshow("heatmap", heatmap)
	# cv2.waitKey()

	phone_ip = "10.0.0.25"
	url = "http://" + phone_ip + "/live?type=some.mp4"

	cap = cv2.VideoCapture(url)

	prev_frame = None
	prev_corners = None

	while cap.isOpened():
		ret, frame = cap.read()

		if ret:
			# st_time = time.time()
			lines, corners = board_locator.find_chessboard(frame, lattice_model, prev=(prev_frame, prev_corners))
			# print("Found board in {} s".format(time.time() - st_time))

			disp = frame.copy()
			for corner in corners:
				cv2.circle(disp, corner, 5, (255, 0, 0), 3)

			if prev_frame is not None:
				grid = get_color_diff_grid(frame, prev_frame, corners, prev_corners)
				heatmap = color_diff_display(frame, corners, grid)
				cv2.imshow("heatmap", cv2.resize(heatmap, None, fx=0.75, fy=0.75))

			cv2.imshow("disp", cv2.resize(disp, None, fx=0.75, fy=0.75))
			cv2.waitKey(1)

			prev_frame = frame
			prev_corners = corners


