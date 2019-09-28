import board_segmentation
from datetime import datetime
import cv2
import sys
import os
import numpy as np


def mark_point(event, x, y, flags, params):
	global corners
	if event == cv2.EVENT_LBUTTONDOWN:
		print("Marked: {}, {}".format(x, y))
		corners.append((x, y))


file = sys.argv[1]
img = cv2.imread(file)

save_dir = "squares"
corners = []

while True:
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", mark_point)

	while True:
		corners = []
		cv2.imshow("image", img)

		c = chr(cv2.waitKey())

		if c == "y":
			break

	cv2.destroyWindow("image")

	disp = img.copy()

	for i in range(4):
		cv2.line(disp, corners[i], corners[(i+1)%4], (255, 0, 0), 2)

	cv2.imshow("check", disp)
	c = chr(cv2.waitKey())

	if c == "y":
		break

chunks = board_segmentation.segment_board(img, corners)

now = datetime.now()

for i in range(len(chunks)):
	corners, center = chunks[i]
	bottom = np.max(corners[:, 1])
	top = bottom - 150
	if top < 0:
		top = 0
	left = np.min(corners[:, 0])
	right = np.max(corners[:, 0])

	subimg = img[int(top):int(bottom), int(left):int(right)]
	cv2.imwrite(os.path.join(save_dir, now.strftime("square-%Y%m%d%H%M%S-{:0>2d}.jpg".format(i))), subimg)