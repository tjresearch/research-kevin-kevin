import cv2
import os
import line_detection
import board_locator

def dist(p1, p2):
	return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)



def mouse_event(event, x, y, flags, params):
	global img, intersections

	if event == cv2.EVENT_LBUTTONDOWN:
		closest_point = None
		closest_dist = img.shape[0] + img.shape[1]
		for point in intersections:
			cur_dist = dist((x, y), point)
			if cur_dist < closest_dist:
				closest_point = point
				closest_dist = cur_dist

		if


path = "images/chessimgs930/IMG_7837.jpg"
img = cv2.imread(path)

model = board_locator.load_model("models/lattice_points_model.json", "models/lattice_points_model.h5")

lines = line_detection.find_lines_rho_theta(img)
intersections = {p:0 for p in board_locator.get_intersections(lines) if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]}



cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_event)

cv2.imshow("image", img)
cv2.waitKey()
