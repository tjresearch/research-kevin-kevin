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
    print("Usage: python3 video_feed.py phone_ip_addr")

FPS = cap.get(5) #ipcamera app locks this at 25
resolution = (int(cap.get(4)), int(cap.get(3))) #vert to horiz res
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
print("FPS", FPS)
print("resolution", resolution)
out = cv2.VideoWriter("./assets/output.avi", fourcc, FPS, resolution)

while cap.isOpened(): #Joshua's doc says while True but I think that's a typo
    ret, frame = cap.read()
    frame = cv2.flip(cv2.transpose(frame), 0)
    if ret:
        cv2.imshow("Feed", frame)
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
out.release()
cv2.destroyAllWindows()
