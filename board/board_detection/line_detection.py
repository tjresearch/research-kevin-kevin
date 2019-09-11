import cv2
import numpy as np

CLAHE_PARAMS = [[3, (2, 6), 5],
				[3, (6, 2), 5],
				[5, (3, 3), 5],
				[0, (0, 0), 0]]

def canny(img, sigma=0.25):
	"""Applies Canny edge detection to the given image."""

	v = np.median(img)

	img = cv2.medianBlur(img, 5)
	img = cv2.GaussianBlur(img, (7, 7), 2)

	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))

	return cv2.Canny(img, lower, upper)

def line_detector(img):
	"""Applies probabilistic Hough transform to the edge gradient."""

	out = []
	lines = cv2.HoughLinesP(img, 1, np.pi/180, 40, minLineLength=50, maxLineGap=15)

	if lines is None:
		return []

	for line in np.reshape(lines, (-1, 4)):
		out += [[[int(line[0]), int(line[1])], [int(line[2]), int(line[3])]]]

	return out

def clahe(img, limit, grid, iters): # Taken from Czyzewski, et al.
	"""Applies CLAHE with 3 sets of hyperparameters."""

	for i in range(iters):
		img = cv2.createCLAHE(limit, grid).apply(img)

	if limit != 0:
		kernel = np.ones((10, 10), dtype=np.uint8)
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	return img

img = cv2.imread("images/chessboard1.jpg")

img_height = 720
img_width = 1280

img = cv2.resize(img, (img_width, img_height))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

i = 0
lines = []

for arr in CLAHE_PARAMS:
	temp = clahe(gray, limit=arr[0], grid=arr[1], iters=arr[2])
	lines += list(line_detector(canny(gray)))

for line in lines:
	cv2.line(img, tuple(line[0]), tuple(line[1]), (255, 0, 0), 2)

cv2.imshow("board", img)
cv2.waitKey()