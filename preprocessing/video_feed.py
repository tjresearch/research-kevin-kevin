import sys
import cv2

phone_ip = sys.argv[1]
url = "http://" + phone_ip + "/live?type=some.mp4"
print(url)
cap = cv2.VideoCapture(url)
if (cap.isOpened()== False):
    print("Usage: python3 video_feed.py phone_ip_addr")

# frame_ct = 0
while cap.isOpened(): #Joshua's doc says while True but I think that's a typo
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Feed", frame)
        if cv2.waitKey(33) == 27: #esc to quit
            break
        # frame_ct += 1
