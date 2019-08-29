import sys, termios
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
    print("input phone_ip")
    phone_ip = input().strip()
    url = "http://" + phone_ip + "/live?type=some.mp4"
    # print(url)
    cap = cv2.VideoCapture(url)
    while True:
        key = get_key()
        if key == '\x1b': #escape to quit
            return
        if key == '\x20': #space to take pic
            print("space")
            ret, img = cap.read()
            cv2.imshow("Frame "+frame_ct, img)

if __name__ == '__main__':
    main()
