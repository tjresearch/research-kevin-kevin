import cv2
import os
import numpy as np
import board_locator
import line_detection
import board_segmentation

filename = "chessboard2.jpg"
img = cv2.imread(os.path.join("images", filename))

# img = cv2.resize(img, (1280, 720))

# lines = line_detection.find_lines_rho_theta(img)
# hor, vert = board_locator.separate_lines(lines)
# line_disp = img.copy()
#
# for line in lines:
# 	rho, theta = line
# 	a = np.cos(theta)
# 	b = np.sin(theta)
# 	x0 = a * rho
# 	y0 = b * rho
# 	x1 = int(x0 + 1000 * -b)
# 	y1 = int(y0 + 1000 * a)
# 	x2 = int(x0 - 1000 * -b)
# 	y2 = int(y0 - 1000 * a)
# 	if line in hor:
# 		cv2.line(line_disp, (x1, y1), (x2, y2), (255, 0, 0), 2)
# 	elif line in vert:
# 		cv2.line(line_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
# 	else:
# 		cv2.line(line_disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
# cv2.imshow("lines", line_disp)
#
# cv2.waitKey()

lattice_point_model = board_locator.load_model("models/lattice_points_model.json", "models/lattice_points_model.h5")
print("Loaded model.")
lines, corners = board_locator.find_chessboard(img, lattice_point_model)
