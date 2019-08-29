import sys
import cv2

phone_ip = sys.argv[1]
url = "http://" + phone_ip + "/live?type=some.mp4"
# print(url)
cap = cv2.VideoCapture(url)
ret = True
frame_ct = 0
while ret: #Joshua's doc says while True but I think that's a typo
    ret, img = cap.read()
    cv2.imshow("Frame", frame_ct, img)
    frame_ct += 1
