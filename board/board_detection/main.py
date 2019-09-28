import cv2
import os
import numpy as np
import board_locator
import board_segmentation

filename = "chessboard4.jpg"
img = cv2.imread(os.path.join("images", filename))

lines = board_locator.find_lines(img)
line_disp = img.copy()

for line in lines:
	rho, theta = line
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a * rho
	y0 = b * rho
	x1 = int(x0 + 1000 * -b)
	y1 = int(y0 + 1000 * a)
	x2 = int(x0 - 1000 * -b)
	y2 = int(y0 - 1000 * a)
	cv2.line(line_disp, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow("lines", line_disp)

edges = board_locator.find_chessboard(img)
chunks = board_segmentation.segment_board_from_edges(img, edges)

chunk_disp = img.copy()

for chunk in chunks:
	corners, center = chunk
	top = np.min(corners[:, 1])
	bottom = np.max(corners[:, 1])
	left = np.min(corners[:, 0])
	right = np.max(corners[:, 0])
	box = ((left, bottom), (right, bottom), (right, bottom - 150), (left, bottom - 150))
	for i in range(4):
		cv2.line(chunk_disp, (int(box[i][0]), int(box[i][1])), (int(box[(i+1)%4][0]), int(box[(i+1)%4][1])), (255, 0, 0), 2)

cv2.imshow("chunks", chunk_disp)

cv2.waitKey()