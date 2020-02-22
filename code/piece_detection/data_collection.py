"""
script to label pieces given an input image of a chessboard
1. click on corners of board
2. board is split into squares

for each square:
1. label color (w/b)
2. label piece (r/n/b/q/k/p)
	(space to skip 1 and 2)
3. space to confirm
4. square saved as {sq_num}-{piece_in_SAN}.jpg
	(in a subfolder of the given output_dir)

uses orthophoto, canny, pose-estimation
"""

from datetime import datetime
import cv2
import os
import numpy as np
import sys, time

from identify_pieces import split_chessboard, order_points

sys.path.insert(1, '../board_detection')
import board_locator, board_segmentation

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
display chessboard image
run board_locator's find_chessboard method
if board_locator wrong, allow user to click on four corners of board
return list of corners
"""
def find_board(img, lattice_point_model):
	cv2.namedWindow("full_image")
	global corners

	cv2.imshow("full_image", img)

	disp = img.copy()

	st_locate_time = time.time()
	# lines, corners = board_locator.find_chessboard(img, lattice_point_model)
	print("Located board in {} s".format(time.time() - st_locate_time))
	corners = []
	for corner in corners:
		cv2.circle(disp, (int(corner[0]), int(corner[1])), 3, (255, 0, 0), 2)
	cv2.imshow("full_image", disp)

	c = chr(cv2.waitKey())
	if c != " ":
		#manual override on corners from board_locator
		#reselect corners of board to segment
		corners = []
		cv2.imshow("full_image", img)

		print("ESC to quit")
		while True:
			cv2.setMouseCallback("full_image", mark_point)

			while True:
				cv2.imshow("full_image", img)
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

			for corner in corners:
				cv2.circle(disp, (int(corner[0]), int(corner[1])), 3, (255, 0, 0), 2)

			cv2.imshow("full_image", disp)
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
	return corners

"""
show image of square, get label, save to save_dir
converted to take the list of squares imgs and their indices
(as given by split_chessboard)
"""
def label_subimgs(img, squares, indices, file, save_dir):
	cv2.namedWindow("subimg", cv2.WINDOW_NORMAL)
	for i in range(len(squares)):
		img = squares[i]
		indx = indices[i]

		cv2.imshow("subimg", img)

		#get piece label
		piece_label = file+"_subimg_"+str(indx)
		piece = piece_label_handler(piece_label)

		#save
		filename = "{}-{}.jpg".format(indx, piece)
		full_path = os.path.join(save_dir, filename)
		cv2.imwrite(full_path, img)
		print("sqr_{} saved to {}\n".format(indx, full_path))

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
for given file,
	segment board into squares
	use orthophoto to identify poss pieces
	use projectPoints to estimate piece height
	show piece, get user-inputted label, save to dir
(saves in-place)
"""
def save_squares(file, outer_dir, lattice_point_model):
	#setup file IO
	save_dir = os.path.join(outer_dir, file[file.rfind("/")+1:file.rfind(".")])
	os.mkdir(save_dir)
	print("output dir: {}".format(save_dir))
	img = cv2.imread(file)

	#find corners of board
	corners = find_board(img, lattice_point_model)

	#take corners, split image into subimgs of viable squares & their indicies
	squares, indices = split_chessboard(img, corners)
	# print(len(squares))
	# print(indices)

	#label squares with pieces, save
	label_subimgs(img, squares, indices, file, save_dir)

"""
for each file in input img_dir_path,
	make output dir for labelled squares
	call save_squares()
"""
def main():
	if len(sys.argv) < 3:
		print("usage: python data_collection.py input_dir output_dir")
		print("-> input_dir (of imgs), output_dir (to store subimgs)")
		exit(0)

	print("Loading board model...")
	model_dir = "../models"
	st_load_time = time.time()
	lattice_point_model = board_locator.load_model(os.path.join(model_dir, "lattice_points_model.json"),
												   os.path.join(model_dir, "lattice_points_model.h5"))
	print("Loaded in {} s".format(time.time() - st_load_time))

	global corners
	img_dir_path = sys.argv[1]

	#make dir of current time for subimgs
	now = datetime.now()
	today_dir = now.strftime("%Y%m%d%H%M%S")
	head = sys.argv[2] #output dir
	save_dir = os.path.join(head, today_dir)
	os.mkdir(save_dir)
	print("save dir: {}".format(save_dir))

	ct = 0
	print(len(os.listdir(img_dir_path)))

	#save squares of each file
	for file in os.listdir(img_dir_path):
		ct += 1
		print("img {}/{}".format(ct, len(os.listdir(img_dir_path))))
		if file.startswith("*"): continue #skip if marked as done
		if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
			filepath = os.path.join(img_dir_path, file)
			print("file: {}".format(filepath))
			save_squares(filepath, save_dir, lattice_point_model)
			corners = [] #clear for next board
			print("file {} done".format(filepath))
			os.rename(filepath, os.path.join(img_dir_path, "*{}".format(file))) #mark as done

	#mark whole dir as done
	last_dir_i = img_dir_path[0:len(img_dir_path)-1].rfind("/")
	os.rename(img_dir_path, os.path.join(img_dir_path[:last_dir_i], "*{}".format(img_dir_path[last_dir_i+1:]))) #mark as done

if __name__ == '__main__':
	main()
