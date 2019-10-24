import board_segmentation
from datetime import datetime
import cv2
import sys
import os
import numpy as np

corners = []
def mark_point(event, x, y, flags, params):
	global corners
	if event == cv2.EVENT_LBUTTONDOWN:
		print("Marked: {}, {}".format(x, y))
		corners.append((x, y))

# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
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

"""not used"""
def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def homography_transform(img, pts, dims):
	src_pts = order_points(pts)
	#fixed, hardcoded height/width
	dst_pts = np.float32([
		[0, 0],
		[dims[0]-1, 0],
		[dims[0]-1, dims[1]-1],
		[0, dims[1]-1],
	])

	H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	print(H)
	# out = cv2.perspectiveTransform(src_pts, H)
	# print(out)

	warped = cv2.warpPerspective(img, H, dims)
	return warped

def find_board(img):
	global corners
	#select corners of board to segment
	print("ESC to quit")
	while True:
		cv2.namedWindow("image")
		cv2.setMouseCallback("image", mark_point)
		print(len(corners))
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
	cv2.destroyWindow("image")

def label_subimgs(img, chunks, file, save_dir):
	region_bounds = []
	#label subimgs
	for i in range(len(chunks)):
		corners, center, region = chunks[i]

		region_bounds.append(region)
		continue

		bottom = np.max(corners[:, 1])
		top = bottom - 150
		if top < 0:
			top = 0
		left = np.min(corners[:, 0])
		right = np.max(corners[:, 0])

		subimg = img[int(top):int(bottom), int(left):int(right)]

		window_name = file+"_subimg_"+str(i)
		cv2.namedWindow(window_name)

		while True:
			cv2.imshow(window_name, subimg)

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

		if color == "w":
			piece = piece.upper()

		#save each subimg by SAN
		#lowercase for black, uppercase for white
		# N = KNIGHT !!!
		filename = "{}_{}.jpg".format(i, piece)
		full_path = os.path.join(save_dir, filename)
		print(full_path)
		cv2.imwrite(full_path, subimg)
		print("subimg_{} saved to {}\n".format(i, full_path))

	disp = img.copy()
	for region in region_bounds:
		r = order_points(region)
		print(r)
		for i in range(4):
			pt_1 = (int(r[i][0]), int(r[i][1]))
			pt_2 = (int(r[(i+1)%4][0]), int(r[(i+1)%4][1]))
			print(pt_1, pt_2)
			cv2.line(disp, pt_1, pt_2, (0, 255, 0), 2)
	while True:
		cv2.imshow("regions", disp)
		cv2.waitKey()
	"""
	take region_subimgs, analyze color
	kmeans?
	"""

def save_squares(file, outer_dir):
	global corners

	save_dir = os.path.join(outer_dir, file[file.rfind("/")+1:file.rfind(".")])
	os.mkdir(save_dir)
	print("output dir: {}".format(save_dir))
	img = cv2.imread(file)

	#fill and order global list corners
	find_board(img)
	#warp for piece detection
	"""
	warp_dims = (400, 400) #r, c
	print(warp_dims)
	if warp_dims[0]%8 or warp_dims[1]%8:
		exit("warp dims not div by 8")
	warped = homography_transform(img, corners, warp_dims)
	cv2.imshow("warped", warped)
	target = ortho_corners(warp_dims)
	for r in target:
		print(r)
	"""

	#chunks have to go in consistent order for this to work
	chunks, H = board_segmentation.roi_segment_board(img, corners)
	label_subimgs(img, chunks, file, save_dir)

def main():
	global corners
	img_dir = sys.argv[1]
	#make dir of current time for subimgs
	now = datetime.now()
	today_dir = now.strftime("%Y%m%d%H%M%S")
	head = sys.argv[2] #"squares"
	save_dir = os.path.join(head, today_dir)
	os.mkdir(save_dir)
	print("save dir: {}".format(save_dir))

	for file in os.listdir(img_dir):
		if file.endswith(".jpg") or file.endswith(".jpeg"):
			filepath = os.path.join(img_dir, file)
			print("file: {}".format(filepath))
			save_squares(filepath, save_dir)
			corners = [] #clear for next board

if __name__ == '__main__':
	main()
