import sys
import cv2

phone_ip = sys.argv[1]
url = "http://" + phone_ip + "/live?type=some.mp4"
print(url)
cap = cv2.VideoCapture(url)
if (cap.isOpened()== False):
    print("Usage: python3 video_feed.py phone_ip_addr")

FPS = cap.get(5)
resolution = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, FPS, resolution)

while cap.isOpened(): #Joshua's doc says while True but I think that's a typo
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Feed", frame)
        out.write(frame)
        cv2.waitKey(1)
            # break
        # frame_ct += 1
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
