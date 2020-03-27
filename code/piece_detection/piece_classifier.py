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
def local_load_model(net_path):
	if os.path.isdir(net_path):
		print("directory given, full net_path required")
		return None
	net = load_model(net_path)
	return net

"""
predict squares, given segmented and orthophoto-pared
"""
def pred_squares(TARGET_SIZE, net, squares, indices, flat_poss=None, graphics_IO=None):
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
		for i in range(len(squares)):
			poss_sets.append(flat_poss[indices[i]])

	#preprocess images, flatten into stack for resnet
	input_stack = []
	for img in squares:
		#convert img to numpy array, preprocess for resnet
		resized_img = cv2.resize(img, dsize=(TARGET_SIZE[1],TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST)
		x = preprocess_input(resized_img)
		input_stack.append(x)
	input_stack = np.array(input_stack)

	#predict on full stack of inputs
	preds = net.predict(input_stack)

	#for ui representation of confidence intervals
	if graphics_IO:
		subfolder = os.path.join(graphics_IO[1], "conf_intervals")
		if not os.path.exists(subfolder):
			os.mkdir(subfolder)
		else:
			for file in os.listdir(subfolder):
				os.remove(os.path.join(subfolder, file))

	#feed preds through poss set checks, adjust preds as needed
	#get SAN and fill pred_board
	pred_board = ["-" for i in range(64)] #flattened 8x8 chessboard
	for i in range(len(preds)):
		pred = preds[i].argsort()[::-1] #most to least likely classes, based on pred
		if poss_sets:
			poss = poss_sets[i]
		else:
			poss = None
		ptr = 0
		pred_SAN = CLASS_TO_SAN[ALL_CLASSES[pred[ptr]]]

		#move down prediction list if prediction is impossible (by chess logic)
		if poss:
			while pred_SAN not in poss:
				if ptr >= len(preds):
					pred_SAN = "?"
				ptr += 1
				pred_SAN = CLASS_TO_SAN[ALL_CLASSES[pred[ptr]]]
				print(ptr, pred_SAN)

		pred_board[indices[i]] = pred_SAN

	#for ui representation of confidence intervals
	# TODO: add poss set checking to visualization
	if graphics_IO:
		for i in range(len(preds)):
			raw_conf = list(preds[i])
			pred = preds[i].argsort()[::-1] #most to least likely classes, based on pred
			piece_map = {raw_conf[i]:ALL_CLASSES[i] for i in range(len(raw_conf))}

			arrow = cv2.imread(os.path.join(graphics_IO[0], "arrow_blank.png"))
			small_sz = (112, 224)

			sqr_img = squares[i]
			disp_sqr = sqr_img.copy()
			disp_sqr = cv2.resize(disp_sqr, dsize=small_sz, interpolation=cv2.INTER_CUBIC)
			ds_shape = (disp_sqr.shape[1], disp_sqr.shape[0])

			piece_filename = "{}.png".format(piece_map[raw_conf[pred[0]]])
			piece_icon = cv2.imread(os.path.join(graphics_IO[0], "piece_images", piece_filename))
			pc_shape = (piece_icon.shape[1], piece_icon.shape[0])

			l_st = (150, 200)
			r_st = (150, 600)
			arrow[l_st[0]:l_st[0]+ds_shape[1],l_st[1]:l_st[1]+ds_shape[0]] = disp_sqr
			arrow[r_st[0]:r_st[0]+pc_shape[1],r_st[1]:r_st[1]+pc_shape[0]] = piece_icon

			disp_conf = []
			for indx in pred:
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

	#rotate board for std display (white on bottom), this is fastest way
	pred_board = np.asarray(pred_board)
	pred_board = np.resize(pred_board, (8,8))
	pred_board = np.rot90(pred_board)
	board = [[None for j in range(8)] for i in range(8)]
	for i in range(8):
		for j in range(8):
			board[i][j] = str(pred_board[i][j])

	return board #return nested lists

"""
classify pieces in img given these: board corners, piece_nnet, TARGET_SIZE of nnet
optional arg: prev state--in same form as output of this method (array of ltrs)
"""
def classify_pieces(img, corners, net, TARGET_SIZE, prev_state=None, graphics_IO=None):
	squares, indices, ortho_guesses = split_chessboard(img, corners, graphics_IO)

	#compute possible next moves from prev state, flatten to 1D list
	flat_poss = []
	if prev_state:
		stacked_poss = get_stacked_poss(prev_state)

		#matching same orientation as board, ASSUMES WHITE ON LEFT OF FRAME
		for c in range(8):
			for r in range(7, -1, -1):
				flat_poss.append(stacked_poss[r][c])

		# print("flat poss, pic oriented")
		# for r in range(8):
		# 	for c in range(8):
		# 		print(flat_poss[r*8+c], end=', ')
		# 	print()

	board = pred_squares(TARGET_SIZE, net, squares, indices, flat_poss=flat_poss, graphics_IO=graphics_IO)
	return board, ortho_guesses
