import sys, time
import os
import numpy as np
import cv2

sys.path.insert(1, '../board_detection')
from board_segmentation import regioned_segment_board

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

# https://stackoverflow.com/questions/19363293/whats-the-fastest-way-to-increase-color-image-contrast-with-opencv-in-python-c
def increase_color_contrast(src, clim, tgs=(8,8)):
	clahe = cv2.createCLAHE(clipLimit=clim, tileGridSize=tgs) #get CLAHE from normal img

	lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
	l, a, b = cv2.split(lab)  # split on 3 different channels

	l2 = clahe.apply(l)  # apply CLAHE to the L-channel
	lab = cv2.merge((l2,a,b))  # merge channels

	output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert back to BGR
	return output

"""
transform Canny edge version of chessboard
identify possible squares w/ pieces based on # of canny pixels in square
return 8x8 binary np array
	piece = 1, empty = 0
"""
def get_ortho_guesses(src, top_ortho_regions, H, SQ_SIZE):
	high_contrast = increase_color_contrast(src, 3.5)

	#same as canny() in line_detection.py but no lower hysteresis thresh
	#and no medianBlur, to find black pieces
	sigma = 0.25
	v = np.median(high_contrast)
	lower = 0
	upper = int(min(255, (1.0 + sigma) * v))
	canny_img = cv2.Canny(high_contrast, lower, upper)

	#get topdown projection of Canny
	dims = (SQ_SIZE*8, SQ_SIZE*8)
	topdown = cv2.transpose(cv2.warpPerspective(canny_img, H, dims))

	#identify number of significant canny points based on white_pix_thresh
	canny_cts = []
	white_pix_thresh = topdown[topdown!=0].mean() #take upper half of canny pix
	# white_pix_thresh = topdown.mean() #take upper half of ALL pix
	for reg in top_ortho_regions: #bounds are transposed
		ct = 0
		for c in range(reg[0][1], reg[1][1]):
			for r in range(reg[0][0], reg[3][0]):
				if topdown[r][c] > white_pix_thresh: #thresh set above
					ct += 1
		canny_cts.append(ct)

	#identify squares that pass threshold for possibly having a piece
	#aiming for perfect recall (mark all pieces at expense of accuracy)
	mark_thresh = SQ_SIZE #pixel threshold to mark current sqr as significant
	flat_piece_binary = [0 for i in range(64)]
	for i in range(64):
		cc = canny_cts[i]
		if cc > mark_thresh:
			flat_piece_binary[i] = 1
		else:
			flat_piece_binary[i] = 0

	piece_binary = np.asarray(flat_piece_binary).reshape(-1, 8)
	return piece_binary

"""
use solvePnPRansac, projectPoints on 9x9 array of sqr intersections
to estimate piece height in pixels given piece height in sqrs
return estimated height for every square of board
"""
def estimate_bounds(src, sqr_corners, piece_height, graphics_IO=None):
	#get imgpts of chessboard intersections
	full_brd_corners = []	#left to right, top to bottom
	for r in range(8):
		sqrs = sqr_corners[r*8:(r+1)*8]
		for sq in sqrs:
			full_brd_corners.append([sq[0][0],sq[0][1]])
		full_brd_corners.append([sqrs[-1][1][0],sqrs[-1][1][1]])
	last_row = sqr_corners[-8:]
	for sq in last_row:
		full_brd_corners.append([sq[3][0], sq[3][1]])
	full_brd_corners.append([last_row[-1][2][0],last_row[-1][2][1]])
	full_brd_corners = np.asarray(full_brd_corners) #81x2

	#81x2 of coords (0,0) -> (9,9)
	objp = np.zeros((81,3), np.float32)
	coords = np.mgrid[0:9,0:9].T
	coords[:,:,[1,0]] = coords[:,:,[0,1]]
	objp[:,:2] = coords.reshape(-1,2)

	#solvePnPRansac with full_brd_corners and objp
	src_r, src_c = src.shape[:-1]
	camera_matrix = np.asarray([[src_c, 0, src_c/2],[0, src_c, src_r/2],[0, 0, 1]])
	dist_coeffs = np.zeros((4,1))
	retval, rvec, tvec, inliers = cv2.solvePnPRansac(objp, full_brd_corners, camera_matrix, dist_coeffs)

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

	#for bounding boxes ui, find full 3D bounds
	if graphics_IO:
		disp_bounds = []
		for r in range(8):
			for c in range(8):
				offset = 0.1
				#points of interest: front pts of base, front pts of top
				POI = [[r+1-offset,c,piece_height],[r+1-offset,c+1,piece_height],[r+1-offset,c+1,0],[r+1-offset,c,0],
						[r-offset,c,piece_height],[r-offset,c+1,piece_height],[r-offset,c+1,0],[r-offset,c,0]]
				for pt in POI:
					disp_bounds.append(pt)
		disp_bounds = np.asarray(disp_bounds).astype(np.float32)

		disp_bounds_proj, jac = cv2.projectPoints(disp_bounds, rvec, tvec, camera_matrix, dist_coeffs)
		disp_bounds_proj = disp_bounds_proj.astype(int)

		disp_pix_bounds = []
		for i in range(0, len(disp_bounds_proj), 8):
			my_group = []
			for shift in range(8):
				my_group.append(tuple(disp_bounds_proj[i+shift][0]))
			disp_pix_bounds.append(my_group)

		disp = src.copy()
		for bound in disp_pix_bounds:
			for i in range(4): #draw front face
				cv2.line(disp, bound[i%4], bound[(i+1)%4], (255,0,0), 1) #teal: (255,195,0)
			for i in range(4): #back face
				cv2.line(disp, bound[i%4+4], bound[(i+1)%4+4], (255,0,0), 1)
			for i in range(4): #front face to back
				cv2.line(disp, bound[i], bound[i+4], (255,0,0), 1)

		cv2.imwrite(os.path.join(graphics_IO[1], "bounding_boxes.jpg"), disp)

	return pix_bounds

def corners_to_imgs(src, sqr_corners, ortho_guesses, TARGET_SIZE, graphics_IO=None):
	imgs = []
	indices = []
	piece_height = TARGET_SIZE[0]//TARGET_SIZE[1] #2 squares tall
	bounds = estimate_bounds(src, sqr_corners, piece_height, graphics_IO)

	#for orthophoto ui img
	subfolder = None
	if graphics_IO:
		# https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
		overlay = src.copy()
		disp = src.copy()
		alpha = 0.40

		for i in range(len(sqr_corners)):
			if not ortho_guesses[i]: continue
			corners = sqr_corners[i].astype(int) #cw from top-left
			cv2.fillConvexPoly(overlay, corners, (50,200,255))

		cv2.addWeighted(overlay, alpha, disp, 1-alpha, 0, disp)
		cv2.imwrite(os.path.join(graphics_IO[1], "orthophoto_guesses.jpg"), disp)

		#for unsheared_sqrs below
		subfolder = os.path.join(graphics_IO[1], "unsheared_sqrs")
		if not os.path.exists(subfolder):
			os.mkdir(subfolder)
		else:
			for file in os.listdir(subfolder):
				os.remove(os.path.join(subfolder, file))

	#crop square out of full image
	for i in range(len(sqr_corners)):
		if not ortho_guesses[i]: continue

		corners = sqr_corners[i] #cw from top-left
		shear_box = bounds[i]

		#perspective transform to normalize parallelogram to rectangle
		dims = (TARGET_SIZE[1], TARGET_SIZE[0])
		dst_box = [(0,0), (dims[0],0), dims, (0,dims[1])] #cw, xy origin top-left
		H, _ = cv2.findHomography(np.array(shear_box), np.array(dst_box))
		unshear = cv2.warpPerspective(src, H, dims)

		#add rectangle to img list
		imgs.append(unshear)
		indices.append(i)

		#for ui
		if graphics_IO:
			arrow = cv2.imread(os.path.join(graphics_IO[0], "arrow_blank.png"))

			small_sz = (TARGET_SIZE[1], TARGET_SIZE[0])
			disp_unshear = unshear.copy()
			disp_unshear = cv2.resize(disp_unshear, dsize=small_sz, interpolation=cv2.INTER_CUBIC)

			#get outer bounding box
			tl_r = min([p[1] for p in shear_box])
			tl_c = min([p[0] for p in shear_box])
			tr_r = max([p[1] for p in shear_box])
			tr_c = max([p[0] for p in shear_box])

			para = src[max(tl_r,0):tr_r,max(tl_c,0):tr_c]
			para = cv2.resize(para, dsize=small_sz, interpolation=cv2.INTER_CUBIC)

			l_st = (150, 150)
			r_st = (150, 700)
			arrow[l_st[0]:l_st[0]+small_sz[1],l_st[1]:l_st[1]+small_sz[0]] = para
			arrow[r_st[0]:r_st[0]+small_sz[1],r_st[1]:r_st[1]+small_sz[0]] = disp_unshear

			cv2.imwrite(os.path.join(subfolder, "sq_{}.jpg".format(i)), arrow)

	return imgs, indices

"""
for given chessboard image and corners of board...
1. segments board into squares
2. uses orthophoto to identify poss pieces
3. uses projectPoints to estimate piece height

returns (list of individual sqr imgs, ortho_guesses)
"""
def split_chessboard(src, brd_corners, TARGET_SIZE, graphics_IO=None):
	#downsize large resolutions
	scale_to = (960, 720)
	old_shape = src.shape
	if src.size > scale_to[0]*scale_to[1]:
		src = ResizeWithAspectRatio(src, width=scale_to[1])
		for i in range(4):
			brd_corners[i] = (int(brd_corners[i][0] * scale_to[1] / old_shape[1]), int(brd_corners[i][1] * scale_to[1] / old_shape[1]))

	brd_corners = order_points(brd_corners)

	#segment board
	sqr_corners, top_ortho_regions, H = regioned_segment_board(src, brd_corners, TARGET_SIZE[1], graphics_IO)

	#use orthophoto to find poss piece locations
	ortho_guesses = get_ortho_guesses(src, top_ortho_regions, H, TARGET_SIZE[1])
	ortho_guesses = ortho_guesses.flatten()

	#turn corner coords into list of imgs
	return (*corners_to_imgs(src, sqr_corners, ortho_guesses, TARGET_SIZE, graphics_IO), ortho_guesses)
