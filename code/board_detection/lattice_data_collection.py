import cv2
import os

def mouse_event(event, x, y, flags, params):
	global img

	if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:

		print("{}, {}".format(x, y))
		subimg = img[y - 10:y + 11, x - 10:x + 11]

		# subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
		# subimg = cv2.threshold(subimg, 0, 255, cv2.THRESH_OTSU)[1]
		# subimg = cv2.Canny(subimg, 0, 255)

		cv2.imshow("sub", subimg)

		if event == cv2.EVENT_LBUTTONDOWN:
			save_dir = "images/lattice_points/yes"
		else:
			save_dir = "images/lattice_points/no"

		file_id = "%03d.jpg" % len(os.listdir(save_dir))

		cv2.imwrite(os.path.join(save_dir, file_id), subimg)


path = "images/chessimgs930/IMG_7837.jpg"
img = cv2.imread(path)

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_event)

cv2.imshow("image", img)
cv2.waitKey()


