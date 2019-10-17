import sys, time
import cv2
import board_locator
import numpy as np

"""
shows live input of ipcamera feed (use hotspot)
esc to quit
"""

phone_ip = sys.argv[1]
url = "http://" + phone_ip + "/live?type=some.mp4"
print(url)
cap = cv2.VideoCapture(url)
if not cap.isOpened():
	print("Usage: python3 video_feed.py phone_ip_addr")

FPS = cap.get(5) #ipcamera app locks this at 25
resolution = (int(cap.get(4)), int(cap.get(3))) #vert to horiz res
print("FPS", FPS)
print("resolution", resolution)

while cap.isOpened(): #Joshua's doc says while True but I think that's a typo
	ret, frame = cap.read()
	frame = cv2.flip(cv2.transpose(frame), 0)
	if ret:
		lines = board_locator.find_lines_improved(frame)
		line_disp = frame.copy()
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

		cv2.imshow("Lines", line_disp)
		# cv2.imshow("Feed", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == 27: #esc
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()
