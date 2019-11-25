from datetime import datetime
import cv2
import os
import numpy as np
import sys
sys.path.insert(1, '../board_detection')
import board_segmentation #from /board_detection


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
def find_poss_pieces(img, region_bounds, H, SQ_SIZE):
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

"""
show image of square, get label, save to save_dir
"""
def label_subimgs(img, square_bounds, poss_pieces, tops, file, save_dir):
	for i in range(len(square_bounds)):
		if not poss_pieces[i]: continue

		#segment full image into square
		corners = square_bounds[i]
		bottom = np.max(corners[:, 1])
		top = tops[i][1] if tops[i][1] > 0 else 0
		left = np.min(corners[:, 0])
		right = np.max(corners[:, 0])
		subimg = img[int(top):int(bottom), int(left):int(right)]

		#get piece label
		window_name = file+"_subimg_"+str(i)
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
		cv2.imshow(window_name, subimg)
		piece = piece_label_handler(window_name)

		#save
		filename = "{}-{}.jpg".format(i, piece)
		full_path = os.path.join(save_dir, filename)
		cv2.imwrite(full_path, subimg)
		print("subimg_{} saved to {}\n".format(i, full_path))

"""
return SAN of input piece and color
"""
def piece_label_handler(window_name):
	while True:
		color = "?"
		print("select color")
		c = chr(cv2.waitKey())
		if c in {"w", "b"}:
			color = c
		elif c in {" ", "e"}:
			color = "e"
		elif c == "\x1b":
			exit("escaped")

		print("select piece")
		piece = ""
		if color != "e":
			piece = "?"
			c = chr(cv2.waitKey())
			if c in {"p", "q", "r", "n", "k", "b", "e"}:
				piece = c
			elif c == "\x1b":
				exit("escaped")

		print("space to confirm, any other to redo")
		if piece:
			print("color: {}\npiece: {}".format(color, piece))
		else:
			print("blank square")

		if chr(cv2.waitKey()) == " ":
			break
		elif c == "\x1b":
			exit("escaped")
		else:
			print("redo")

	cv2.destroyWindow(window_name)

	#convert input color+piece to SAN
	if color == "w":
		piece = piece.upper()
	if not piece: #blank sq
		piece = "x"

	return piece

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
for given file,
	segment board into squares
	use orthophoto to identify poss pieces
	use projectPoints to estimate piece height
	show piece, get user-inputted label, save to dir
"""
def save_squares(file, outer_dir):
	global corners

	#setup file IO
	save_dir = os.path.join(outer_dir, file[file.rfind("/")+1:file.rfind(".")])
	os.mkdir(save_dir)
	print("output dir: {}".format(save_dir))
	img = cv2.imread(file)

	#downsize large resolutions
	scale_to = (960, 720)
	if img.size > scale_to[0]*scale_to[1]:
		img = ResizeWithAspectRatio(img, width=scale_to[1])

	#fill and order global list corners
	find_board(img)

	#segment board
	SQ_SIZE = 100
	chunks, H = board_segmentation.regioned_segment_board(img, corners, SQ_SIZE)

	"""
	chunks[0] = corners (squares defined by four corners)
	chunks[1] = centers (squares defined by four corners)
	chunks[2] = region_bounds (search regions of orthophoto, defined by four corners)
	"""
	region_bounds = [c[2] for c in chunks]

	#use orthophoto to find poss piece locations
	poss_pieces = find_poss_pieces(img, region_bounds, H, SQ_SIZE)
	print(poss_pieces)
	poss_pieces = poss_pieces.flatten()

	piece_height = 2 #squares tall
	square_bounds = [c[0] for c in chunks]
	tops = estimate_tops(img, piece_height, square_bounds)

	#label squares with pieces, save
	label_subimgs(img, square_bounds, poss_pieces, tops, file, save_dir)

"""
for each file in input img_dir,
	make output dir for labelled squares
	call save_squares()
"""
def main():
	if len(sys.argv) < 3:
		print("usage: python data_collection.py input_dir output_dir")
		print("-> input_dir (of imgs), output_dir (to store subimgs)")
		exit(0)

	global corners
	img_dir = sys.argv[1]

	#make dir of current time for subimgs
	now = datetime.now()
	today_dir = now.strftime("%Y%m%d%H%M%S")
	head = sys.argv[2] #output dir
	save_dir = os.path.join(head, today_dir)
	os.mkdir(save_dir)
	print("save dir: {}".format(save_dir))

	#save squares of each file
	for file in os.listdir(img_dir):
		if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
			filepath = os.path.join(img_dir, file)
			print("file: {}".format(filepath))
			save_squares(filepath, save_dir)
			corners = [] #clear for next board
			print("file {} done".format(filepath))

if __name__ == '__main__':
	main()
