"""
putting a piece-recognition nnet with the
piece splitting system in piece_labelling.py (modified)

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
from piece_labelling import ResizeWithAspectRatio

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
print(ALL_CLASSES)

#because of global array in piece_labelling, have to copy-paste methods from piece_labelling
"""from piece_labelling"""
"""
mouse callback for find_board()
"""
corners = []
def mark_point(event, x, y, flags, params):
	global corners
	if event == cv2.EVENT_LBUTTONDOWN:
		print("Marked: {}, {}".format(x, y))
		corners.append((x, y))

"""
order four points in top-left, top-right, bottom-left, bottom-right order
return np array of points
https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
"""
def order_points(pts):
	if type(pts) == list:
		pts = np.array(pts)

	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

"""
display chessboard image, allow user to click on four corners of board
to segment board into squares
"""
def find_board(img):
	global corners
	#select corners of board to segment
	print("ESC to quit")
	while True:
		cv2.namedWindow("image")
		cv2.setMouseCallback("image", mark_point)

		while True:
			cv2.imshow("image", img)
			print("pick four corners, space to finish, any other to redo")

			c = chr(cv2.waitKey())
			if c == " ":
				break
			elif c == "\x1b":
				exit("escaped")
			else:
				corners = []
				print("corners cleared")

		disp = img.copy()
		corners = order_points(corners)

		for i in range(4): #will crash if < 4 corners marked
			cv2.line(disp, tuple(corners[i]), tuple(corners[(i+1)%4]), (255, 0, 0), 2)

		cv2.imshow("image", disp)
		print("space to confirm board, any other to redo")

		c = chr(cv2.waitKey())
		if c == " ":
			break
		elif c == "\x1b":
			exit("escaped")
		else:
			corners = []
			print("corners cleared")
	# cv2.destroyWindow("image")

"""
transform Canny edge version of chessboard
identify possible squares w/ pieces based on # of canny pixels in square
return 8x8 binary np array
	piece = 1, empty = 0
"""
def get_ortho_guesses(img, region_bounds, H, SQ_SIZE):
	dims = (SQ_SIZE*8, SQ_SIZE*8)

	#same as canny() in line_detection.py but
	#no lower hysteresis thresh and no medianBlur
	#to find black pieces
	sigma = 0.25
	v = np.median(img)
	lower = 0
	upper = int(min(255, (1.0 + sigma) * v))

	canny_edge_img = cv2.Canny(img, lower, upper)
	narr = np.asarray(canny_edge_img[:,:])

	#get topdown projection of Canny
	topdown = cv2.transpose(cv2.warpPerspective(canny_edge_img, H, dims))

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
	canny_ct_thresh = 10 #aiming for perfect recall (mark all pieces at expense of accuracy)
	piece_binary = np.asarray([1 if n > canny_ct_thresh else 0 for n in canny_cts]).reshape(-1, 8)

	return piece_binary

"""
use solvePnPRansac, projectPoints on 9x9 array of sqr intersections
to estimate piece height
return estimated height for every square of board
"""
def estimate_tops(img, piece_height, square_bounds):
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

	#find centers of each square
	to_draw = []
	for r in range(8):
		for c in range(8):
			# for pt in [[r,c,0],[r,c+1,0],[r+1,c,0],[r,c,1]]:
			for pt in [[r+0.5,c+0.5,0],[r+0.5,c+0.5,piece_height]]:
				to_draw.append(pt)
	to_draw = np.asarray(to_draw).astype(np.float32)

	#use centers and Ransac to project piece heights in image
	proj_pts, jac = cv2.projectPoints(to_draw, rvec, tvec, camera_matrix, dist_coeffs)
	proj_pts = proj_pts.astype(int)
	tops = []
	for i in range(1,len(proj_pts),2):
		tops.append(proj_pts[i][0])
	tops = np.asarray(tops)

	return tops
"""end from piece_labelling"""

def corners_to_imgs(img, square_bounds, poss_pieces, tops):
	imgs = []
	indices = []
	for i in range(len(square_bounds)):
		if not poss_pieces[i]: continue

		#segment full image into square
		corners = square_bounds[i]
		bottom = np.max(corners[:, 1])
		top = tops[i][1] if tops[i][1] > 0 else 0
		left = np.min(corners[:, 0])
		right = np.max(corners[:, 0])
		subimg = img[int(top):int(bottom), int(left):int(right)]
		imgs.append(subimg)
		indices.append(i)

	return imgs, indices

"""
for given file,
	segment board into squares
	use orthophoto to identify poss pieces
	use projectPoints to estimate piece height
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

	#fill and order global list corners
	# find_board(img)

	corners = order_points(corners)

	# disp = img.copy()
	# for corner in corners:
	# 	cv2.circle(disp, corner, 3, (255, 0, 0), 2)
	# cv2.imshow("corners", disp)
	# cv2.waitKey()

	#segment board
	SQ_SIZE = 100
	chunks, H = board_segmentation.regioned_segment_board(img, corners, SQ_SIZE)

	# for chunk in chunks:
	# 	center = chunk[1]
	# 	cv2.circle(disp, (int(center[0]), int(center[1])), 3, (255, 0, 0), 2)
	#
	# cv2.imshow("centers", disp)
	# cv2.waitKey()

	"""
	chunks[0] = corners (squares defined by four corners)
	chunks[1] = centers (squares defined by four corners)
	chunks[2] = region_bounds (search regions of orthophoto, defined by four corners)
	"""
	region_bounds = [c[2] for c in chunks]

	#use orthophoto to find poss piece locations
	ortho_guesses = get_ortho_guesses(img, region_bounds, H, SQ_SIZE)
	print("ortho_guesses:")
	print(ortho_guesses)
	ortho_guesses = ortho_guesses.flatten()

	piece_height = 2 #squares tall
	square_bounds = [c[0] for c in chunks]

	# for bounds in square_bounds:
	# 	disp = img.copy()
	# 	for corner in bounds:
	# 		cv2.circle(disp, (int(corner[0]), int(corner[1])), 3, (255, 0, 0), 2)
	# 	cv2.imshow("disp", disp)
	# 	cv2.waitKey()

	tops = estimate_tops(img, piece_height, square_bounds)

	#turn corner coords into list of imgs
	return corners_to_imgs(img, square_bounds, ortho_guesses, tops)

def local_load_model(net_path):
	#load model
	if os.path.isdir(net_path):
		net_file = sorted(os.listdir(net_path))[-1] #lowest alphabetically = highest acc
		net_path = os.path.join(net_path, net_file)
	# print("Loading:",net_path)
	# st_load_time = time.time()
	net = load_model(net_path)
	# load_time = time.time()-st_load_time
	# print("\nLoaded {} in {} s.".format(net_path, round(load_time, 3)))
	return net

#not verbose
def pred_squares(TARGET_SIZE, net, squares, indices, flat_poss=None):
	print("pred_squares start")
	global CLASS_TO_SAN, ALL_CLASSES

	#predict squares
	st_pred_time = time.time()
	sq_preds = ["-" for i in range(64)] #8x8 chessboard

	for sq_i in range(len(squares)):
		img = squares[sq_i]
		indx = indices[sq_i]

		poss = {}
		if flat_poss:
			poss = flat_poss[indx] #not sq_i, since full flat_poss passed
			if len(poss) == 1: #skip pred if only one possible piece there
				sq_preds[indx] = poss.pop()
				print("unique sqr:", sq_preds[indx])
				continue

		#convert img to numpy array, preprocess for resnet
		resized_img = cv2.resize(img, dsize=(TARGET_SIZE[1],TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST)
		x = preprocess_input(resized_img)
		x = np.expand_dims(x, axis=0) #need to add dim to put into resnet

		#get prediction from resnet, get SAN from prediction
		preds = net.predict(x)[0].argsort()[::-1] #most to least likely predictions
		ptr = 0
		pred_SAN = CLASS_TO_SAN[ALL_CLASSES[preds[ptr]]]

		#move down prediction list if prediction is impossible (by chess logic)
		if poss:
			while pred_SAN not in poss:
				print("collision of pred:", pred_SAN)
				print("poss_set:", poss)
				print("moving down list:", preds)
				if ptr >= len(preds):
					print("ran out of predictions")
					pred_SAN = "?"
				ptr += 1
				pred_SAN = CLASS_TO_SAN[ALL_CLASSES[preds[ptr]]]
				print(ptr, pred_SAN)

		sq_preds[indx] = pred_SAN
		cv2.destroyWindow("{}".format(indx))

	#rotate board for std display (white on bottom)
	sq_preds = np.asarray(sq_preds)
	sq_preds = np.resize(sq_preds, (8,8))
	sq_preds = np.rot90(sq_preds)

	board = [[None for j in range(8)] for i in range(8)]
	for i in range(8):
		for j in range(8):
			board[i][j] = str(sq_preds[i][j])

	pred_time = time.time()-st_pred_time
	print("\nPrediction time: {} s.".format(round(pred_time, 3)))

	return board

#prev state given in same form as output of this method (array of ltrs)
def classify_pieces(img, corners, net, TARGET_SIZE, prev_state=None):
	squares, indices = split_chessboard(img, corners)
	# print(indices)

	#compute possible next moves from prev state, flatten to 1D list,
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

def main():
	if len(sys.argv)<3:
		print("usage: python identify_pieces.py [model_path]|[model_dir] [chessboard_img] [verbose=False]")
		exit("\t->model_dir will pick highest acc model in dir")

	VERBOSE = sys.argv[3] if len(sys.argv)==4 else False

	# model_filename = "exp_hybrid_allclass_9896_1579640093_highangle_RES152V2.h5"
	net = local_load_model(sys.argv[1])

	#split img into squares
	img_path = sys.argv[2]
	print("Img src:", img_path)
	img = cv2.imread(img_path)
	squares, indices = split_chessboard(img, corners)
	print(len(squares))

	#run each square through nnet
	TARGET_SIZE = (224,112)

	#predict squares
	board = pred_squares(TARGET_SIZE, net, squares, indices)
	display(board)

	if not VERBOSE:
		cv2.destroyAllWindows()
		exit("verbose output not enabled")

	#predict squares (verbose)
	st_vpred_time = time.time()
	sq_vpred_outputs = [(np.array([]), None, None, None) for i in range(64)] #8x8 chessboard
	for i in range(len(squares)):
		img = squares[i]
		indx = indices[i]

		resized_img = cv2.resize(img, dsize=(TARGET_SIZE[1],TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST)
		x = preprocess_input(resized_img)
		x = np.expand_dims(x, axis=0) #need to add dim to put into resnet

		preds = net.predict(x)[0]
		top_inds = preds.argsort()[::-1]

		sq_vpred_outputs[indx] = (img, resized_img, preds, top_inds)

	vpred_time = time.time()-st_vpred_time
	print("\nVerbose prediction time: {} s.".format(round(vpred_time, 3)))

	#show verbose predictions
	cv2.namedWindow("original", cv2.WINDOW_NORMAL)
	cv2.namedWindow("resized", cv2.WINDOW_NORMAL)
	for img, resized_img, preds, top_inds in sq_vpred_outputs:
		if img.size == 0: continue
		cv2.imshow("original", img)
		cv2.imshow("resized", resized_img)
		for i in top_inds[:5]: #only show top five
			print('{}: {}'.format(ALL_CLASSES[i], preds[i]))
		print()
		cv2.waitKey()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
