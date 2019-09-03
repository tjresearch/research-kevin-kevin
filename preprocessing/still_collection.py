"""
cv2.waitKey() doesn't like the live video
"""

import sys, termios, time
import cv2

def get_key():
    old = termios.tcgetattr(sys.stdin)
    new = termios.tcgetattr(sys.stdin)
    new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
    new[6][termios.VMIN] = 1
    new[6][termios.VTIME] = 0
    termios.tcsetattr(sys.stdin, termios.TCSANOW, new)
    key = None
    try:
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, old)
    return key

def main():
    phone_ip = sys.argv[1]
    url = "http://" + phone_ip + "/live?type=some.mp4"
    print(url)
    cap = cv2.VideoCapture(url)
    if (cap.isOpened()== False):
        print("Usage: python3 video_feed.py phone.ip.addr")

    while cap.isOpened(): #Joshua's doc says while True but I think that's a typo
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): #doesn't work
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
