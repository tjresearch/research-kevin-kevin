import cv2
import os
import numpy as np
import board_locator
import line_detection
import board_segmentation

filename = "chessboard1.jpg"
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
corners = board_locator.find_chessboard(img, lattice_point_model)

corner_disp = img.copy()

for i in range(4):
	cv2.line(corner_disp, (int(corners[i][0]), int(corners[i][1])), (int(corners[(i+1)%4][0]), int(corners[(i+1)%4][1])), (255, 0, 0), 2)

cv2.imshow("corners", corner_disp)

chunks = board_segmentation.segment_board(img, corners)

chunk_disp = img.copy()
composite_disp = img.copy()

for chunk in chunks:
	corners, center = chunk
	top = np.min(corners[:, 1])
	bottom = np.max(corners[:, 1])
	left = np.min(corners[:, 0])
	right = np.max(corners[:, 0])
	box = ((left, bottom), (right, bottom), (right, bottom - 150), (left, bottom - 150))
	for i in range(4):
		cv2.line(chunk_disp, (int(box[i][0]), int(box[i][1])), (int(box[(i+1)%4][0]), int(box[(i+1)%4][1])), (255, 0, 0), 2)
		cv2.line(composite_disp, (int(corners[i][0]), int(corners[i][1])), (int(corners[(i+1)%4][0]), int(corners[(i+1)%4][1])), (0, 0, 255), 2)

cv2.imshow("composite", composite_disp)
cv2.imshow("chunks", chunk_disp)

cv2.waitKey()