"""
putting a piece-recognition nnet with the
piece splitting system in data_collection.py (modified)

1. wait for model to load (150 s on avg)
2. click corners of board in chessboard_img
3. board will be segmented and pieces identified and labelled
4. (if verbose) probabilities shown
"""

import os, sys
import time
import numpy as np
import cv2

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

sys.path.insert(1, '../board_detection')
import board_segmentation #from /board_detection
sys.path.insert(2, '../chess_logic')
from pgn_helper import display #from /chess_logic
from next_moves import get_stacked_poss

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

"""
resize to width/height while keeping aspect ratio
return resized img
https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
"""
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]

	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))

	return cv2.resize(image, dim, interpolation=inter)

"""
order four points clockwise from top-left corner
return np array of points
https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
"""
def order_points(pts):
	if type(pts) == list:
		pts = np.array(pts)

	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

"""
transform Canny edge version of chessboard
identify possible squares w/ pieces based on # of canny pixels in square
return 8x8 binary np array
	piece = 1, empty = 0
"""
def get_ortho_guesses(img, region_bounds, H, SQ_SIZE):
	dims = (SQ_SIZE*8, SQ_SIZE*8)

	#same as canny() in line_detection.py but no lower hysteresis thresh
	#and no medianBlur, to find black pieces
	sigma = 0.25
	v = np.median(img)
	lower = 0
	upper = int(min(255, (1.0 + sigma) * v))

	canny_edge_img = cv2.Canny(img, lower, upper)
	narr = np.asarray(canny_edge_img[:,:])

	#get topdown projection of Canny
	topdown = cv2.transpose(cv2.warpPerspective(canny_edge_img, H, dims))
	cv2.imshow("topdown", topdown)

	#identify number of significant canny points based on white_pix_thresh
	canny_cts = []
	white_pix_thresh = topdown[topdown!=0].mean() #take upper half of canny pix
	for reg in region_bounds: #bounds are transposed
		subimg = topdown[int(reg[0][0]):int(reg[3][0]), int(reg[0][1]):int(reg[1][1])]

		ct = 0
		for c in range(reg[0][1], reg[1][1]):
			for r in range(reg[0][0], reg[3][0]):
				if topdown[r][c] > white_pix_thresh: #thresh set above
					ct += 1
		canny_cts.append(ct)

	#identify squares that pass threshold for possibly having a piece
	#aiming for perfect recall (mark all pieces at expense of accuracy)
	local_thresh = 5 #thresh to mark current sqr
	# above_thresh = 25 #thresh for the "square below" mark
	flat_piece_binary = [0 for i in range(64)]
	for i in range(64):
		cc = canny_cts[i]
		# if cc > above_thresh and i < 56:
			# flat_piece_binary[i+8] = 1
		if cc > local_thresh:
			flat_piece_binary[i] = 1
		else:
			flat_piece_binary[i] = 0

	piece_binary = np.asarray(flat_piece_binary).reshape(-1, 8)
	return piece_binary

"""
use solvePnPRansac, projectPoints on 9x9 array of sqr intersections
to estimate piece height
return estimated height for every square of board
"""
def estimate_bounds(img, square_bounds, piece_height, graphics_on=False):
	#get imgpts of chessboard intersections
	board_corners = []	#left to right, top to bottom
	for r in range(8):
		sqrs = square_bounds[r*8:(r+1)*8]
		for sq in sqrs:
			board_corners.append([sq[0][0],sq[0][1]])
		board_corners.append([sqrs[-1][1][0],sqrs[-1][1][1]])
	last_row = square_bounds[-8:]
	for sq in last_row:
		board_corners.append([sq[3][0], sq[3][1]])
	board_corners.append([last_row[-1][2][0],last_row[-1][2][1]])
	board_corners = np.asarray(board_corners) #81x2

	#81x2 of coords (0,0) -> (9,9)
	objp = np.zeros((81,3), np.float32)
	coords = np.mgrid[0:9,0:9].T
	coords[:,:,[1,0]] = coords[:,:,[0,1]]
	objp[:,:2] = coords.reshape(-1,2)

	#solvePnPRansac with board_corners and objp
	img_r, img_c = img.shape[:-1]
	camera_matrix = np.asarray([[img_c, 0, img_c/2],[0, img_c, img_r/2],[0, 0, 1]])
	dist_coeffs = np.zeros((4,1))
	retval, rvec, tvec, inliers = cv2.solvePnPRansac(objp, board_corners, camera_matrix, dist_coeffs)

	#find front face of 3D bounding box
	desired_bounds = []
	for r in range(8):
		for c in range(8):
			#points of interest: front pts of base, front pts of top
			POI = [[r+0.75,c,piece_height],[r+0.75,c+1,piece_height],[r+0.75,c+1,0],[r+0.75,c,0]]
			for pt in POI:
				desired_bounds.append(pt)
	desired_bounds = np.asarray(desired_bounds).astype(np.float32)

	#use centers and Ransac to project piece heights in image
	bounds_proj, jac = cv2.projectPoints(desired_bounds, rvec, tvec, camera_matrix, dist_coeffs)
	bounds_proj = bounds_proj.astype(int)

	#reshape to group top coords
	pix_bounds = []
	for i in range(0, len(bounds_proj), 4):
		my_group = []
		for shift in range(4):
			my_group.append(bounds_proj[i+shift][0])
		pix_bounds.append(my_group)

	#for display, find full 3D bounds
	if graphics_on:
		disp_bounds = []
		for r in range(8):
			for c in range(8):
				#points of interest: front pts of base, front pts of top
				POI = [[r+0.75,c,piece_height],[r+0.75,c+1,piece_height],[r+0.75,c+1,0],[r+0.75,c,0],
						[r-0.25,c,piece_height],[r-0.25,c+1,piece_height],[r-0.25,c+1,0],[r-0.25,c,0]]
				for pt in POI:
					disp_bounds.append(pt)
		disp_bounds = np.asarray(disp_bounds).astype(np.float32)

		disp_bounds_proj, jac = cv2.projectPoints(disp_bounds, rvec, tvec, camera_matrix, dist_coeffs)
		disp_bounds_proj = disp_bounds_proj.astype(int)

		disp_pix_bounds = []
		for i in range(0, len(disp_bounds_proj), 8):
			my_group = []
			for shift in range(8):
				my_group.append(disp_bounds_proj[i+shift][0])
			disp_pix_bounds.append(my_group)

		print("work with disp_pix_bounds list to draw bounding boxes")

	return pix_bounds

def corners_to_imgs(img, poss_pieces, square_bounds, piece_height, SQ_SIZE):
	imgs = []
	indices = []
	bounds = estimate_bounds(img, square_bounds, piece_height)

	for i in range(len(square_bounds)):
		if not poss_pieces[i]: continue

		#crop square out of full image
		corners = square_bounds[i] #cw from top-left
		shear_box = bounds[i]

		#perspective transform to normalize parallelogram to rectangle
		pix_height = piece_height*SQ_SIZE
		dims = (SQ_SIZE, pix_height)
		dst_box = [(0,0), (dims[0],0), dims, (0,dims[1])] #cw, xy origin top-left
		H, _ = cv2.findHomography(np.array(shear_box), np.array(dst_box))
		unshear = cv2.warpPerspective(img, H, dims)
		# cv2.imshow("unshear sqr", unshear)
		# cv2.waitKey()

		#add rectangle to img list
		imgs.append(unshear)
		indices.append(i)

	return imgs, indices

"""
for given file, corners of board...
1. segment board into squares
2. use orthophoto to identify poss pieces
3. use projectPoints to estimate piece height

return list of img arrays
"""
def split_chessboard(img, corners):
	#downsize large resolutions
	scale_to = (960, 720)
	old_shape = img.shape
	if img.size > scale_to[0]*scale_to[1]:
		img = ResizeWithAspectRatio(img, width=scale_to[1])
		for i in range(4):
			corners[i] = (int(corners[i][0] * scale_to[1] / old_shape[1]), int(corners[i][1] * scale_to[1] / old_shape[1]))

	corners = order_points(corners)

	#segment board
	SQ_SIZE = 112
	sqr_info, H = board_segmentation.regioned_segment_board(img, corners, SQ_SIZE)

	"""
	for si in sqr_info:
		si[0] = square defined by four corners
		si[1] = search region of orthophoto on sqr, defined by four corners
	"""
	square_bounds = [si[0] for si in sqr_info]
	region_bounds = [si[1] for si in sqr_info]

	#use orthophoto to find poss piece locations
	ortho_guesses = get_ortho_guesses(img, region_bounds, H, SQ_SIZE)
	print("ortho_guesses:")
	print(ortho_guesses)
	ortho_guesses = ortho_guesses.flatten()

	#turn corner coords into list of imgs
	piece_height = 2 #squares tall
	return corners_to_imgs(img, ortho_guesses, square_bounds, piece_height, SQ_SIZE)

#load model
def local_load_model(net_path):
	if os.path.isdir(net_path):
		net_file = sorted(os.listdir(net_path))[-1] #lowest alphabetically = highest acc
		net_path = os.path.join(net_path, net_file)

	net = load_model(net_path)
	return net

"""
predict squares, given segmented and orthophoto-pared
"""
def pred_squares(TARGET_SIZE, net, squares, indices, flat_poss=None):
	print("pred_squares start")
	global CLASS_TO_SAN, ALL_CLASSES
	st_pred_time = time.time()

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

	#feed preds through poss set checks, repred as needed
	#get SAN and fill pred_board
	pred_board = ["-" for i in range(64)] #flattened 8x8 chessboard

	for i in range(len(preds)):
		pred = preds[i].argsort()[::-1] #most to least likely classes, based on pred
		poss = poss_sets[i]
		ptr = 0
		pred_SAN = CLASS_TO_SAN[ALL_CLASSES[pred[ptr]]]

		#move down prediction list if prediction is impossible (by chess logic)
		if poss:
			while pred_SAN not in poss:
				print("collision of pred:", pred_SAN)
				print("poss_set:", poss)
				print("moving down list:", pred)
				if ptr >= len(preds):
					print("ran out of predictions")
					pred_SAN = "?"
				ptr += 1
				pred_SAN = CLASS_TO_SAN[ALL_CLASSES[pred[ptr]]]
				print(ptr, pred_SAN)

		pred_board[indices[i]] = pred_SAN

	#rotate board for std display (white on bottom)
	# TODO: rot90 w/out converting to numpy and back
	pred_board = np.asarray(pred_board)
	pred_board = np.resize(pred_board, (8,8))
	pred_board = np.rot90(pred_board)

	#convert from numpy to nested lists
	board = [[None for j in range(8)] for i in range(8)]
	for i in range(8):
		for j in range(8):
			board[i][j] = str(pred_board[i][j])

	#print time
	pred_time = time.time()-st_pred_time
	print("\nPrediction time: {} s.".format(round(pred_time, 3)))

	return board #return nested lists

"""
classify pieces in img given these: board corners, piece_nnet, TARGET_SIZE of nnet
optional arg: prev state--in same form as output of this method (array of ltrs)
"""
def classify_pieces(img, corners, net, TARGET_SIZE, prev_state=None):
	squares, indices = split_chessboard(img, corners)

	#compute possible next moves from prev state, flatten to 1D list
	flat_poss = []
	if prev_state:
		stacked_poss = get_stacked_poss(prev_state)

		#matching same orientation as board
		for c in range(8):
			for r in range(7, -1, -1):
				flat_poss.append(stacked_poss[r][c])

		print("flat poss, pic oriented")
		for r in range(8):
			for c in range(8):
				print(flat_poss[r*8+c], end=', ')
			print()

	board = pred_squares(TARGET_SIZE, net, squares, indices, flat_poss)
	return board

if __name__ == '__main__':
	print("no main method")
