import cv2
import os
import line_detection
import board_locator

def dist(p1, p2):
	return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)

def update_display(img, intersections, window_name):
	disp = img.copy()
	for point in intersections:
		if intersections[point]:
			cv2.circle(disp, (int(point[0]), int(point[1])), 3, (0, 255, 0), 2)
	cv2.imshow(window_name, disp)

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

		intersections[closest_point] = not intersections[closest_point]
		update_display(img, intersections, "image")


path = "images/chessimgs930/IMG_7841.jpg"
img = cv2.imread(path)

model = board_locator.load_model("models/lattice_points_model.json", "models/lattice_points_model.h5")

lines = line_detection.find_lines_rho_theta(img)
intersections = {p:board_locator.validate_lattice_point(model, p, img) for p in board_locator.get_intersections(lines) if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]}

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_event)

update_display(img, intersections, "image")
c = cv2.waitKey()

if c == ord("q"):
	exit()

for point in intersections:
	if 10 < point[0] < img.shape[1] - 10 and 10 < point[1] < img.shape[0] - 10:
		if intersections[point]:
			save_dir = "images/lattice_points/yes"
		else:
			save_dir = "images/lattice_points/no"

		subimg = img[point[1] - 10:point[1] + 11, point[0] - 10:point[0] + 11]

		file_id = "%03d.jpg" % len(os.listdir(save_dir))
		cv2.imwrite(os.path.join(save_dir, file_id), subimg)