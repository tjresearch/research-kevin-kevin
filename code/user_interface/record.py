"""
DEPRECATED
"""

import sys, time
import cv2

"""
shows live input of ipcamera feed (use hotspot)
spacebar to save frame as .jpg
esc to quit
output vid and .jpgs saved to local folder /assets
"""

phone_ip = sys.argv[1]
url = "http://" + phone_ip + "/live?type=some.mp4"
print(url)
cap = cv2.VideoCapture(url)
if (cap.isOpened()== False):
    print("Usage: python video_feed.py phone_ip_addr [record_feed]")

record_bool = sys.argv[2] if len(sys.argv)>2 else False

FPS = cap.get(5) #ipcamera app locks this at 25
resolution = (int(cap.get(4)), int(cap.get(3))) #vert to horiz res
print("FPS", FPS)
print("resolution", resolution)

if record_bool:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter("./assets/output.avi", fourcc, 25, resolution)

while cap.isOpened(): #Joshua's doc says while True but I think that's a typo
    ret, frame = cap.read()
    frame = cv2.flip(cv2.transpose(frame), 0)
    if ret:
        cv2.imshow("Feed", frame)
        if record_bool:
            out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "): #save frame on spacebar
            img_file = "./assets/"+str(time.time())+".jpg"
            cv2.imwrite(img_file, frame)
            print(img_file, "saved")
        elif key == 27: #esc
            break
    else:
        break

cap.release()
if record_bool:
    out.release()
cv2.destroyAllWindows()
