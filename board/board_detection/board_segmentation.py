import board_locator
import cv2
import numpy as np

img = cv2.imread("images/chessboard4.jpg")

lines = board_locator.find_lines(img)
disp = img.copy()

for line in lines:
	print(line)
	rho, theta = line
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a * rho
	y0 = b * rho
	x1 = int(x0 + 1000 * -b)
	y1 = int(y0 + 1000 * a)
	x2 = int(x0 - 1000 * -b)
	y2 = int(y0 - 1000 * a)
	cv2.line(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow("lines", disp)

warped = board_locator.find_chessboard(img)

cv2.imshow("warped", warped)
cv2.waitKey()