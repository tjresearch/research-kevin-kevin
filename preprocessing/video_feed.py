import sys, time
import cv2

"""
shows live input of ipcamera feed (use hotspot)
esc to quit
"""

phone_ip = sys.argv[1]
url = "http://" + phone_ip + "/live?type=some.mp4"
print(url)
cap = cv2.VideoCapture(url)
if (cap.isOpened()== False):
    print("Usage: python3 video_feed.py phone_ip_addr")

FPS = cap.get(5) #ipcamera app locks this at 25
resolution = (int(cap.get(4)), int(cap.get(3))) #vert to horiz res
print("FPS", FPS)
print("resolution", resolution)

while cap.isOpened(): #Joshua's doc says while True but I think that's a typo
    ret, frame = cap.read()
    frame = cv2.flip(cv2.transpose(frame), 0)
    if ret:
        cv2.imshow("Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: #esc
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
