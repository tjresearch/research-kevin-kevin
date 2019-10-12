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

def save_squares(file, outer_dir):
	global corners

	save_dir = os.path.join(outer_dir, file[file.rfind("/")+1:file.rfind(".")])
	os.mkdir(save_dir)
	print("output dir: {}".format(save_dir))
	img = cv2.imread(file)

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

		cv2.destroyWindow("image")

		disp = img.copy()

		for i in range(4):
			cv2.line(disp, corners[i], corners[(i+1)%4], (255, 0, 0), 2)

		cv2.imshow("check", disp)
		print("space to confirm board, any other to redo")

		c = chr(cv2.waitKey())
		if c == " ":
			break
		elif c == "\x1b":
			exit("escaped")
		else:
			corners = []
			print("corners cleared")
		cv2.destroyWindow("check")

	cv2.destroyWindow("check")
	chunks = board_segmentation.segment_board(img, corners)

	#label subimgs
	for i in range(len(chunks)):
		corners, center = chunks[i]
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

def main():
	global corners
	img_dir = sys.argv[1]
	#make dir of current time for subimgs
	now = datetime.now()
	today_dir = now.strftime("%Y%m%d%H%M%S")
	save_dir = os.path.join("squares", today_dir)
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
