"""
piece identification =
square_splitter.py -> piece_classifier.py
"""

import sys, time
import os
import numpy as np
import cv2

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

from square_splitter import split_chessboard

sys.path.insert(1, '../chess_logic')
# from pgn_helper import display
from next_moves import get_stacked_poss

"""
load model
"""
def local_load_model(nnet_path):
	if os.path.isdir(nnet_path):
		print("directory given, full nnet_path required")
		return None
	nnet = load_model(nnet_path)
	return nnet

"""
predict board state, given segmented and orthophoto-pared sqr imgs
"""
def get_pred_board(nnet, TARGET_SIZE, sqr_imgs, indices, flat_poss=None, graphics_IO=None):
	CLASS_TO_SAN = {
		'black_bishop':'b',
		'black_king':'k',
		'black_knight':'n',
		'black_pawn':'p',
		'black_queen':'q',
		'black_rook':'r',
		'empty':'-',
		'white_bishop':'B',
		'white_king':'K',
		'white_knight':'N',
		'white_pawn':'P',
		'white_queen':'Q',
		'white_rook':'R'
	}
	ALL_CLASSES = [*CLASS_TO_SAN.keys()]

	#populate poss sets for given squares
	poss_sets = []
	if flat_poss:
		for i in range(len(sqr_imgs)):
			poss_sets.append(flat_poss[indices[i]])

	#preprocess images, flatten into stack for resnet
	input_stack = []
	for sqr_img in sqr_imgs:
		#convert sqr_img to numpy array, preprocess for resnet
		resized = cv2.resize(sqr_img, dsize=(TARGET_SIZE[1],TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST)
		x = preprocess_input(resized)
		input_stack.append(x)
	input_stack = np.array(input_stack)

	#predict on full stack of inputs
	raw_preds = nnet.predict(input_stack)

	#for ui representation of confidence intervals
	if graphics_IO:
		subfolder = os.path.join(graphics_IO[1], "conf_intervals")
		if not os.path.exists(subfolder):
			os.mkdir(subfolder)
		else:
			for file in os.listdir(subfolder):
				os.remove(os.path.join(subfolder, file))

	#feed raw_preds through poss set checks, adjust raw_preds as needed
	#get SAN and fill pred_board
	pred_board = ["-" for i in range(64)] #flattened 8x8 chessboard
	for i in range(len(raw_preds)):
		cls_preds = raw_preds[i].argsort()[::-1] #most to least likely classes, based on pred
		if poss_sets:
			poss = poss_sets[i]
		else:
			poss = None
		ptr = 0
		pred_SAN = CLASS_TO_SAN[ALL_CLASSES[cls_preds[ptr]]]

		#move down prediction list if prediction is impossible (by chess logic)
		if poss:
			while pred_SAN not in poss:
				if ptr >= len(raw_preds): #no possible pieces, given poss set
					pred_SAN = "?"
				ptr += 1
				pred_SAN = CLASS_TO_SAN[ALL_CLASSES[cls_preds[ptr]]]
				print(ptr, pred_SAN)

		pred_board[indices[i]] = pred_SAN

	#for ui representation of confidence intervals
	# TODO: add poss set checking to visualization
	if graphics_IO:
		for i in range(len(raw_preds)):
			raw_conf = list(raw_preds[i])
			cls_preds = raw_preds[i].argsort()[::-1] #most to least likely classes, based on pred
			piece_map = {raw_conf[i]:ALL_CLASSES[i] for i in range(len(raw_conf))}

			arrow = cv2.imread(os.path.join(graphics_IO[0], "arrow_blank.png"))
			small_sz = (TARGET_SIZE[1], TARGET_SIZE[0])

			sqr_img = sqr_imgs[i]
			disp_sqr = sqr_img.copy()
			disp_sqr = cv2.resize(disp_sqr, dsize=small_sz, interpolation=cv2.INTER_CUBIC)
			ds_shape = (disp_sqr.shape[1], disp_sqr.shape[0])

			piece_filename = "{}.png".format(piece_map[raw_conf[cls_preds[0]]])
			piece_icon = cv2.imread(os.path.join(graphics_IO[0], "piece_images", piece_filename))
			pc_shape = (piece_icon.shape[1], piece_icon.shape[0])

			l_st = (150, 200)
			r_st = (150, 600)
			arrow[l_st[0]:l_st[0]+ds_shape[1],l_st[1]:l_st[1]+ds_shape[0]] = disp_sqr
			arrow[r_st[0]:r_st[0]+pc_shape[1],r_st[1]:r_st[1]+pc_shape[0]] = piece_icon

			disp_conf = []
			for indx in cls_preds:
				conf = raw_conf[indx]
				pc = piece_map[conf]
				disp_conf.append("{}: {}".format(pc, str(round(conf, 3))))

			text_orig = (400, 400)
			for disp_i in range(5):
				line = disp_conf[disp_i]
				# print(line)
				dy = disp_i*30
				cv2.putText(arrow, line, (text_orig[0], text_orig[1]+dy), \
							cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0))

			cv2.imwrite(os.path.join(subfolder, "sq_{}.jpg".format(indices[i])), arrow)

	# board = rotate_board_to_std(pred_board)
	# board = [[pred_board[i][j] for j in range(8)] for i in range(8)]
	pred_board = np.asarray(pred_board)
	pred_board = np.resize(pred_board, (8,8))
	board = [["-" for j in range(8)] for i in range(8)]
	for i in range(8):
		for j in range(8):
			board[i][j] = str(pred_board[i][j])
	return board #nested lists
"""
rotate board so white pieces on bottom
"""
def rotate_board_to_std(pred_board, white_on_left):
	#rotate board for std display (white on bottom), this is fastest way
	pred_board = np.asarray(pred_board)
	pred_board = np.resize(pred_board, (8,8))
	pred_board = np.rot90(pred_board)
	if not white_on_left:
		pred_board = np.rot90(np.rot90(pred_board))
	board = [[None for j in range(8)] for i in range(8)]
	for i in range(8):
		for j in range(8):
			board[i][j] = str(pred_board[i][j])
	return board

"""
classify pieces in src img given: board_corners, piece_nnet, TARGET_SIZE of nnet
optional arg: prev state--in same form as output of this method (array of ltrs)
"""
def classify_pieces(src, board_corners, nnet, TARGET_SIZE, white_on_left, prev_state=None, graphics_IO=None):
	sqr_imgs, indices, ortho_guesses = split_chessboard(src, board_corners, TARGET_SIZE, graphics_IO)

	#compute possible next moves from prev state, flatten to 1D list
	flat_poss = []
	#requires white_on_left to exist too
	if prev_state:
		stacked_poss = get_stacked_poss(prev_state)

		#matching same orientation as board
		#!! ASSUMES WHITE ON LEFT OF FRAME
		for c in range(8):
			for r in range(7, -1, -1):
				flat_poss.append(stacked_poss[r][c])

	pred_board = get_pred_board(nnet, TARGET_SIZE, sqr_imgs, indices, flat_poss=flat_poss, graphics_IO=graphics_IO)

	if white_on_left == None:
		white_on_left = find_white_on_left(pred_board)

	return pred_board, ortho_guesses, white_on_left

"""
figure out which side of frame white pieces are on
"""
def find_white_on_left(pred_board):
	#[left side, right side]
	white_count = [0,0]
	black_count = [0,0]
	for r in range(8):
		for c in range(4):
			ltr = pred_board[r][c]
			if ltr != "-":
				if ltr.isupper():
					white_count[0] += 1
				else:
					black_count[0] += 1
		for c2 in range(4, 8):
			ltr = pred_board[r][c2]
			if ltr != "-":
				if ltr.isupper():
					white_count[1] += 1
				else:
					black_count[1] += 1

	more_white_left = white_count[0] > black_count[0]
	more_white_right = white_count[1] > black_count[1]

	print(white_count)
	print(black_count)

	if more_white_left and not more_white_right:
		return True
	elif more_white_right and not more_white_left:
		return False
	else:
		#if more white pieces than black pieces on both sides of the board,
		#or more black than white on both sides
		#return which side has more white pieces overall as a guess
		return more_white_left > more_white_right
