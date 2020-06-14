import numpy as np
import cv2
import os
import board_locator
import board_segmentation
import utils
import sys
from threading import Thread

sys.path.insert(1, "../piece_detection")
import piece_classifier, square_splitter

sys.path.insert(2, "../chess_logic")
from pgn_helper import display
from pgn_writer import find_pgn_move

sys.path.insert(3, "../user_interface")
from query_diagram import diagram_from_board_string

TARGET_SIZE = (224, 112)

SQ_SIZE = 100
dst_size = SQ_SIZE * 8
dst_points = [(SQ_SIZE, SQ_SIZE), (SQ_SIZE, dst_size - SQ_SIZE), (dst_size - SQ_SIZE, dst_size - SQ_SIZE), (dst_size - SQ_SIZE, SQ_SIZE)]

prev_raw_frame = None
prev_frame = None
prev_corners = None
prev_grid = None

cur_calm_streak = 0
good_calm_streak = 15

cur_noise_streak = 0
min_noise_streak = 10

first_calm = True
last_calm_raw_frame = None
last_calm_frame = None
last_calm_corners = None
idx = 0

cur_board = None
new_board = None

cur_diagram = None

white_on_left = None

disp_scale = 0.5

white_on_left = None

pgn_moves = []

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

def save_frame(frame, save_dir, corners):
	filename = "{:03d}.jpg".format(len(os.listdir(save_dir))-1)

	#downsize large resolutions
	scale_to = (720, 960)
	old_shape = frame.shape
	scaled_corners = []
	to_save = None
	if frame.size > scale_to[0]*scale_to[1]:
		to_save = square_splitter.ResizeWithAspectRatio(frame, width=scale_to[1])
		for i in range(4):
			scaled_corners.append((int(corners[i][0] * scale_to[1] / old_shape[1]), int(corners[i][1] * scale_to[1] / old_shape[1])))
	else:
		to_save = frame

	#add CLAHE
	to_save = piece_classifier.increase_color_contrast(to_save, 3.5, (8,8)) #increase color contrast of original

	#save frame
	cv2.imwrite(os.path.join(save_dir, filename), to_save)
	print(" as {}".format(filename)) #print announcement

	#save corners to cached_corners.txt in save_dir
	with open(os.path.join(save_dir, 'cached_corners.txt'), 'a+') as cache_file:
		cache_file.write("{} - {}\n".format(filename, str(scaled_corners)))
		cache_file.flush()
	#
	# disp = to_save.copy()
	# for sc in scaled_corners:
	# 	cv2.circle(disp, (int(sc[0]), int(sc[1])), 3, (255, 0, 0), 2)
	# cv2.imshow("disp", disp)
	# cv2.waitKey()

def rotate_rc_coords(coords, white_on_left):
	rotated = []
	for coord in coords:
		if white_on_left:
			rotated.append([7 - coord[1], coord[0]])
		else:
			rotated.append([coord[1], 7 - coord[0]])
	return rotated

def process_frame(frame, corners, piece_model, calm_comparison=None):
	global new_board, white_on_left, cur_board
	if calm_comparison is not None:
		peaks = np.argwhere(calm_comparison > np.median(calm_comparison) + np.std(calm_comparison) * 0.5).tolist()
		flat_peaks = []

		for i in range(len(peaks)):
			flat_peaks.append(peaks[i][0] * 8 + peaks[i][1])
		board_mask, ortho_guesses, white_on_left = piece_classifier.classify_pieces(frame, corners, piece_model, TARGET_SIZE,
																			   prev_state=cur_board, white_on_left=white_on_left, squares_to_process=flat_peaks)

		rotated_peaks = rotate_rc_coords(peaks, white_on_left)

		for peak in rotated_peaks:
			new_board[peak[0]][peak[1]] = board_mask[peak[0]][peak[1]]
	else:
		new_board, ortho_guesses, white_on_left = piece_classifier.classify_pieces(frame, corners, piece_model, TARGET_SIZE)

def update_calm(raw_frame, frame, corners, piece_model, show_process, save_dir=None, calm_comparison=None):
	global first_calm, last_calm_raw_frame, last_calm_frame, last_calm_corners
	last_calm_raw_frame = raw_frame
	last_calm_frame = frame
	last_calm_corners = corners
	# if show_process:
	# 	cv2.imshow("last_calm", cv2.resize(last_calm_frame, None, fx=disp_scale, fy=disp_scale))
	if first_calm:
		if save_dir:
			print("Saved frame {}".format(idx), end="")
			# if show_process:
			# 	cv2.imshow("last_saved", cv2.resize(raw_frame, None, fx=disp_scale, fy=disp_scale))
			save_frame(raw_frame, save_dir, corners)
		else:
			classify_thread = Thread(target=lambda : process_frame(raw_frame, corners, piece_model, calm_comparison=calm_comparison))
			classify_thread.daemon = True
			classify_thread.start()
		first_calm = False

def process_video_frame(raw_frame, lattice_model, piece_model, show_process, save_dir=None):
	global prev_raw_frame, prev_frame, prev_corners, prev_grid,\
		cur_calm_streak, good_calm_streak, first_calm, last_calm_raw_frame, \
		last_calm_frame, last_calm_corners, idx, cur_noise_streak, min_noise_streak

	frame = piece_classifier.increase_color_contrast(raw_frame, 2, (8, 8))
	if show_process:
		disp = frame.copy()

	if prev_grid is not None and idx - 2 >= 0 and np.max(prev_grid - np.median(prev_grid)) > 5:
		corners = prev_corners
		if cur_calm_streak > good_calm_streak:
			cur_noise_streak = 0
		cur_calm_streak = 0
		cur_noise_streak += 1
	else:
		if last_calm_frame is None:
			_, corners = board_locator.find_chessboard(raw_frame, lattice_model, prev=(prev_raw_frame, prev_corners))

			if cur_calm_streak >= good_calm_streak:
				update_calm(raw_frame, frame, corners, piece_model, show_process, save_dir)
		else:
			if cur_calm_streak >= good_calm_streak:
				if cur_calm_streak == good_calm_streak and cur_noise_streak > min_noise_streak:
					first_calm = True
				calm_comparison = get_color_diff_grid(prev_frame, last_calm_frame, prev_corners, last_calm_corners)
				# if show_process:
				# 	cv2.imshow("calm_grid", cv2.resize(color_diff_display(prev_frame, prev_corners, calm_comparison), None, fx=0.5, fy=0.5))
				dist_from_avg = calm_comparison - np.median(calm_comparison)

				# if the color change grid has less than 7 outliers; 6 is the maximum number of significant changes for a single move
				if len(np.argwhere(calm_comparison > np.median(calm_comparison) + np.std(calm_comparison) * 1.25)) < 7:
					lines, corners = board_locator.find_chessboard(raw_frame, lattice_model, prev=(last_calm_raw_frame, last_calm_corners))
					update_calm(raw_frame, frame, corners, piece_model, show_process, save_dir, calm_comparison)
				else:
					corners = prev_corners
			else:
				first_calm = False
				corners = prev_corners
		cur_calm_streak += 1

	# print("Found board in {} s".format(time.time() - st_time))

	if show_process:
		for corner in corners:
			cv2.circle(disp, corner, 3, (255, 0, 0), 2)

	if idx - 1 >= 0:
		grid = get_color_diff_grid(frame, prev_frame, corners, prev_corners)
	else:
		grid = None

	prev_raw_frame = raw_frame
	prev_frame = frame
	prev_corners = corners
	prev_grid = grid

	if show_process:
		cv2.imshow("disp", cv2.resize(disp, None, fx=disp_scale, fy=disp_scale))

	idx += 1

def show_diagram():
	global cur_board, cur_diagram
	board_string = "".join("".join(row) for row in cur_board)
	diagram = diagram_from_board_string(board_string).convert("RGB")
	cur_diagram = cv2.cvtColor(np.array(diagram), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
	if len(sys.argv) < 2 or len(sys.argv) > 4:
		print("Usage: video_handler.py [src video/phone_ip] [show process] [save dir]")

	delay = 0

	#handle input args
	cap = None
	if sys.argv[1].endswith(".mp4"):
		print("Video file given.")
		cap = cv2.VideoCapture(sys.argv[1])
		# cap.set(cv2.CAP_PROP_POS_FRAMES, 6242)
	else:
		print("IP given (live video).")
		phone_ip = sys.argv[1]
		url = "http://" + phone_ip + "/live?type=some.mp4"
		print(url)
		cap = cv2.VideoCapture(url)

	save_dir = None
	show_process = False
	if len(sys.argv) == 4:
		save_dir = sys.argv[3]

		save_dir_files = {*os.listdir(save_dir)}-{'.DS_Store'}
		if 'cached_corners.txt' not in save_dir_files:
			print("Creating cached_corners.txt in {}".format(save_dir))
			#create file to save corners in later
			f = open(os.path.join(save_dir, 'cached_corners.txt'), 'w+')
			f.close()

		if len(save_dir_files):
			print("\nWARNING: save_dir not empty!\n")
	if len(sys.argv) >= 3:
		show_process = int(sys.argv[2]) #0 or 1
	print(show_process)

	#load models
	model_path = "../models"
	lattice_model = board_locator.load_model(os.path.join(model_path, "lattice_points_model.json"), os.path.join(model_path, "lattice_points_model.h5"))
	if save_dir:
		piece_model = None
	else:
		import time
		print("Loading piece classification model...")
		st = time.time()
		piece_model = piece_classifier.load_model(os.path.join(model_path, "piece_detection_model.h5"))
		print("Piece detection model loaded in {} seconds.".format(round(time.time()-st), 3))

	while cap.isOpened():
		ret, raw_frame = cap.read()

		if not ret: break
		# print("Frame: {}".format(idx))

		if ret:
			process_video_frame(raw_frame, lattice_model, piece_model, show_process, save_dir)

			if new_board is not None and (cur_board is None or not all(cur_board[i // 8][i % 8] == new_board[i // 8][i % 8] for i in range(64))):
				if cur_board:
					display(new_board) #show result of process_video_frame
					#find pgn notation of move, print full pgn move list
					pgn_moves.append(find_pgn_move(cur_board, new_board))
					for i in range(0, len(pgn_moves)-1, 2):
						print("{}. {} {}".format(i//2+1, pgn_moves[i], pgn_moves[i+1]))
					if len(pgn_moves)%2:
						print("{}. {}".format(len(pgn_moves)//2+1, pgn_moves[-1]))

				#update cur_board, graphical display
				cur_board = [[elem for elem in row] for row in new_board]
				diagram_thread = Thread(target=show_diagram)
				diagram_thread.daemon = True
				diagram_thread.start()

			if cur_diagram is not None:
				cv2.imshow("diagram", cur_diagram)

			c = cv2.waitKey(1 * delay)
			if c == ord(" "):
				delay = (delay + 1) % 2

	#save pgn moves to file
	filename = sys.argv[1]+".pgn" #file goes to same dir as input src of vid, current dir if IP
	print("Saving .pgn file to", filename)
	with open(filename, "w+") as f:
		for i in range(0, len(pgn_moves)-1, 2):
			f.write("{}. {} {}\n".format(i//2+1, pgn_moves[i], pgn_moves[i+1]))
		if len(pgn_moves)%2:
			f.write("{}. {}".format(len(pgn_moves)//2+1, pgn_moves[-1]))
	f.close()
