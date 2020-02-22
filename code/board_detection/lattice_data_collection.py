import cv2
import os
import line_detection
import board_locator

def dist(p1, p2):
	return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)

def update_display(img, lattice_dict, window_name):
	disp = img.copy()
	for point in lattice_dict:
		if lattice_dict[point]:
			cv2.circle(disp, (int(point[0]), int(point[1])), 3, (0, 255, 0), 2)
	cv2.imshow(window_name, disp)

def mouse_event(event, x, y, flags, params):
	img, lattice_dict = params
	if event == cv2.EVENT_LBUTTONDOWN:
		closest_point = None
		closest_dist = img.shape[0] + img.shape[1]
		for point in lattice_dict:
			cur_dist = dist((x, y), point)
			if cur_dist < closest_dist:
				closest_point = point
				closest_dist = cur_dist

		lattice_dict[closest_point] = not lattice_dict[closest_point]
		update_display(img, lattice_dict, "image")


def collect_data(img, model):
	lines = line_detection.find_lines_rho_theta(img)
	intersections = [p for p in board_locator.get_intersections(lines) if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]]

	poss_points, lattice_conf = board_locator.validate_lattice_points(model, intersections, img)

	lattice_dict = {poss_points[i]:lattice_conf[i] for i in range(len(poss_points))}

	cv2.namedWindow("image")
	cv2.setMouseCallback("image", mouse_event, (img, lattice_dict))

	update_display(img, lattice_dict, "image")
	c = cv2.waitKey()

	if c != ord("y"):
		return

	for point in lattice_dict:
		if 10 < point[0] < img.shape[1] - 10 and 10 < point[1] < img.shape[0] - 10:
			if lattice_dict[point]:
				save_dir = "images/lattice_points/yes"
			else:
				save_dir = "images/lattice_points/no"

			subimg = img[point[1] - 10:point[1] + 11, point[0] - 10:point[0] + 11]

			file_id = "%03d.jpg" % len(os.listdir(save_dir))
			cv2.imwrite(os.path.join(save_dir, file_id), subimg)


model = board_locator.load_model("models/lattice_points_model.json", "models/lattice_points_model.h5")

path = "images/chessboard4.jpg"
img = cv2.imread(path)

collect_data(img, model)

# url = "http://28.82.217.175/live?type=some.mp4"
#
# cap = cv2.VideoCapture(url)
# if not cap.isOpened():
# 	print("Could not connect to phone.")
# 	exit(1)
#
# FPS = cap.get(cv2.CAP_PROP_FPS)
# resolution = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
# print("FPS: {}".format(FPS))
# print("Resolution: {}".format(resolution))
#
# while cap.isOpened():
# 	ret, frame = cap.read()
# 	# frame = cv2.flip(cv2.transpose(frame), 0)
# 	if ret:
# 		cv2.imshow("Feed", frame)
# 		c = cv2.waitKey(1)
# 		if c == ord(" "):
# 			disp = frame.copy()
# 			collect_data(disp, model)
# 			cv2.destroyWindow("image")
# 		elif c == 27:
# 			break
# 	else:
# 		break