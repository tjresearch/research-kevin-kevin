import numpy as np
import cv2
import os
import board_locator
import board_segmentation
import utils
import sys

sys.path.insert(1, "../piece_detection")
import square_splitter

SQ_SIZE = 100
dst_size = SQ_SIZE * 8
dst_points = [(SQ_SIZE, SQ_SIZE), (SQ_SIZE, dst_size - SQ_SIZE), (dst_size - SQ_SIZE, dst_size - SQ_SIZE), (dst_size - SQ_SIZE, SQ_SIZE)]

prev_raw_frame = None
prev_frame = None
prev_corners = None
prev_grid = None

cur_calm_streak = 0
good_calm_streak = 10
first_calm = True
last_calm_raw_frame = None
last_calm_frame = None
last_calm_corners = None
idx = 0

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
		cv2.fillConvexPoly(heatmap, int_square, int(diff_grid[r, c] / 50 * 255))
	return heatmap

def save_frame(frame, save_dir):
	filename = "{:03d}.jpg".format(len(os.listdir(save_dir)))
	cv2.imwrite(os.path.join(save_dir, filename), frame)

def update_calm(raw_frame, frame, corners):
	global first_calm, last_calm_raw_frame, last_calm_frame, last_calm_corners
	last_calm_raw_frame = raw_frame
	last_calm_frame = frame
	last_calm_corners = corners
	cv2.imshow("last_calm", cv2.resize(last_calm_frame, None, fx=0.5, fy=0.5))
	if first_calm:
		if save_frames:
			print("Saved frame {}".format(idx))
			save_frame(raw_frame, save_dir)
		else:
			print(idx)
			cv2.waitKey()
		first_calm = False

def process_frame(raw_frame):
	global prev_raw_frame, prev_frame, prev_corners, prev_grid,\
		cur_calm_streak, good_calm_streak, first_calm, last_calm_raw_frame, \
		last_calm_frame, last_calm_corners, idx

	frame = square_splitter.increase_color_contrast(raw_frame, 2.5, (8, 8))
	disp = frame.copy()

	if prev_grid is not None and idx - 2 >= 0 and np.max(prev_grid - np.median(prev_grid)) > 5:
		corners = prev_corners
		cur_calm_streak = 0
	else:
		if last_calm_frame is None:
			lines, corners = board_locator.find_chessboard(raw_frame, lattice_model, prev=(prev_raw_frame, prev_corners))
			if cur_calm_streak >= good_calm_streak:
				update_calm(raw_frame, frame, corners)
		else:
			if cur_calm_streak >= good_calm_streak:
				if cur_calm_streak == good_calm_streak:
					first_calm = True
				calm_comparison = get_color_diff_grid(prev_frame, last_calm_frame, prev_corners, last_calm_corners)
				cv2.imshow("calm_grid", cv2.resize(color_diff_display(prev_frame, prev_corners, calm_comparison), None, fx=0.5, fy=0.5))
				dist_from_avg = calm_comparison - np.median(calm_comparison)
				dist_from_avg[dist_from_avg < 0] = 0
				if len(np.argwhere(dist_from_avg > np.mean(dist_from_avg) + np.std(dist_from_avg) * 1.5)) < 5:
					lines, corners = board_locator.find_chessboard(raw_frame, lattice_model, prev=(last_calm_raw_frame, last_calm_corners))
					update_calm(raw_frame, frame, corners)
				else:
					corners = prev_corners
			else:
				first_calm = False
				corners = prev_corners
		cur_calm_streak += 1

	# print("Found board in {} s".format(time.time() - st_time))

	for corner in corners:
		cv2.circle(disp, corner, 5, (255, 0, 0), 3)

	if idx - 1 >= 0:
		grid = get_color_diff_grid(frame, prev_frame, corners, prev_corners)
	else:
		grid = None

	prev_raw_frame = raw_frame
	prev_frame = frame
	prev_corners = corners
	prev_grid = grid

	cv2.imshow("disp", cv2.resize(disp, None, fx=0.5, fy=0.5))

	idx += 1

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

	# phone_ip = "10.0.0.25"
	# url = "http://" + phone_ip + "/live?type=some.mp4"
	#
	# cap = cv2.VideoCapture(url)

	if len(sys.argv[1:]) != 1 and len(sys.argv[1:]) != 3:
		print("usage: video_handler.py [src video] | [save frames?] [save dir]")

	delay = 0

	cap = cv2.VideoCapture(sys.argv[1])

	if len(sys.argv[1:]) > 1:
		save_frames = bool(sys.argv[2])
		save_dir = sys.argv[3]
	else:
		save_frames = False

	while cap.isOpened():
		ret, raw_frame = cap.read()

		# print("Frame: {}".format(idx))

		if ret:
			process_frame(raw_frame)

			c = cv2.waitKey(1 * delay)
			if c == ord(" "):
				delay = (delay + 1) % 2





