import os
import sys
import cv2
import time

sys.path.insert(1, "../board_detection")
import board_locator

sys.path.insert(2, "../piece_detection")
import piece_classifier

sys.path.insert(3, "../chess_logic")
import pgn_helper

if len(sys.argv) != 4:
	print("usage: python main_shell.py [phone ip] [shared models dir] [img_path]")
	exit(1)

phone_ip = sys.argv[1]
url = "http://" + phone_ip + "/live?type=some.mp4"

model_dir = sys.argv[2]

# Load board models
print("Loading board model...")
st_load_time = time.time()
lattice_point_model = board_locator.load_model(os.path.join(model_dir, "lattice_points_model.json"),
											   os.path.join(model_dir, "lattice_points_model.h5"))
print("Loaded in {} s".format(time.time() - st_load_time))

# Load piece models
print("Loading piece model...")
st_load_time = time.time()
# piece_model = identify_pieces.local_load_model(os.path.join(model_dir, "piece_detection_model.h5"))
piece_model = None
print("Loaded in {} s".format(time.time() - st_load_time))

TARGET_SIZE = (224, 112)

# For single image
cv2.namedWindow("original")

img_path = sys.argv[3]
img = cv2.imread(img_path)
cv2.imshow("original", img)
# cv2.waitKey()

st_locate_time = time.time()
lines, corners = board_locator.find_chessboard(img, lattice_point_model)
print("Located board in {} s".format(time.time() - st_locate_time))

disp = img.copy()
for corner in corners:
	cv2.circle(disp, (int(corner[0]), int(corner[1])), 3, (255, 0, 0), 2)

# cv2.imshow("corners", disp)
# cv2.waitKey()

# prev_state = [['-', '-', '-', '-', '-', '-', '-', '-'],
# 			  ['-', '-', '-', '-', 'N', '-', '-', '-'],
# 			  ['-', '-', '-', '-', '-', '-', '-', '-'],
# 			  ['-', '-', '-', '-', 'R', '-', 'q', 'P'],
# 			  ['-', '-', '-', 'r', '-', '-', '-', '-'],
# 			  ['-', '-', '-', '-', '-', 'R', 'Q', '-'],
# 			  ['-', '-', '-', 'N', '-', '-', '-', '-'],
# 			  ['-', '-', '-', '-', 'B', '-', '-', '-']]
# prev_state = None
# print("prev state:")
# # pgn_helper.display(prev_state)
# print(prev_state)
prev_state = None

"""
#calling twice no work
#somehow the orthophoto is very wrong when run a second time
board = identify_pieces.classify_pieces(img, corners, piece_model, TARGET_SIZE)
print()
print(board)
print()
pgn_helper.display(board)
print("-"*60)
"""

# graphics_IO = ("./assets", "./graphics_out")
# if not os.path.exists(graphics_IO[0]):
# 	print("missing assets folder")
# if not os.path.exists(graphics_IO[1]):
# 	os.mkdir(graphics_IO[1])
# print("Pulling assets from {}".format(graphics_IO[0]))
# print("Saving graphics to {}".format(graphics_IO[1]))
graphics_IO = None

board = piece_classifier.classify_pieces(img, corners, piece_model, TARGET_SIZE, graphics_IO, prev_state)
pgn_helper.display(board)
print()
print(board)
print()

print("any key to close")
cv2.waitKey()
cv2.destroyWindow("disp")
exit(0)

# For live feed
cap = cv2.VideoCapture(url)
if not cap.isOpened():
	print("Could not connect to phone.")
	exit(1)

FPS = cap.get(cv2.CAP_PROP_FPS)
resolution = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("FPS: {}".format(FPS))
print("Resolution: {}".format(resolution))

while cap.isOpened():
	ret, frame = cap.read()
	# frame = cv2.flip(cv2.transpose(frame), 0)
	if ret:
		cv2.imshow("Feed", frame)
		c = cv2.waitKey(1)
		if c == ord(" "):
			disp = frame.copy()

			print("Locating board...")
			st_locate_time = time.time()
			lines, corners = board_locator.find_chessboard(frame, lattice_point_model)
			print("Located board in: {} s".format(time.time() - st_locate_time))

			for corner in corners:
				cv2.circle(disp, (int(corner[0]), int(corner[1])), 3, (255, 0, 0), 2)

			cv2.imshow("disp", disp)
			cv2.waitKey()
			cv2.destroyWindow("disp")

			print("Identifying pieces...")
			st_piece_time = time.time()
			board = identify_pieces.classify_pieces(frame, corners, piece_model, TARGET_SIZE)
			print("Identified pieces in: {} s".format(time.time() - st_piece_time))

			pgn_helper.display(board)

		elif c == 27:
			break
	else:
		break
