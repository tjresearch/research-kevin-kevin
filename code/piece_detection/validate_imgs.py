"""
replaced by format_train_data.py
"""

import sys, time
import cv2
import os
import numpy as np
from piece_labelling import piece_label_handler

"""
turn filename (SAN) into human-readable piece name
"""
def get_human_label(filename):
    dash = filename.index('-')
    dot = filename.index('.')
    sq_num = filename[:dash]
    piece = filename[dash+1:dot]
    ext = filename[dot+1:]

    if piece == "x":
        return sq_num, "empty", ext
    elif piece == "?":
        return sq_num, "?", ext

    label = "black_" if piece.islower() else "white_"

    p = piece.lower()
    if p == "r":
        label += "rook"
    elif p == "n":
        label += "knight"
    elif p == "b":
        label += "bishop"
    elif p == "k":
        label += "king"
    elif p == "q":
        label += "queen"
    elif p == "p":
        label += "pawn"
    return sq_num, label, ext

"""
take folder of folders of images as input
    outer_dir
    |_ img_dir
        |_ 0-x.jpg
        ...
    ...
(as produced by board/board_detection/piece_labelling.py)

then rename image files in place based on user input
"""
def main():
    outer_dir = sys.argv[1]
    for dir in os.listdir(outer_dir):
        img_dir = os.path.join(outer_dir, dir)
        print("img_dir:",img_dir)
        for filename in os.listdir(img_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                filepath = os.path.join(img_dir, filename)
                print("file: {}".format(filepath))
                img = cv2.imread(filepath)

                cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
                cv2.imshow(filename, img)

                sq_num, label, ext = get_human_label(filename)
                print(label)

                print("\nspace if label correct, ESC to quit, any other key to relabel")
                c = chr(cv2.waitKey())
                if c == "\x1b":
                    exit("escaped")
                elif c == " ":
                    cv2.destroyWindow(filename)
                    continue

                print("relabel piece", filename)
                piece = piece_label_handler(filename)

                #rename old file
                new_filename = "{}-{}.{}".format(sq_num, piece, ext)
                print("renaming {} to {}".format(filename, new_filename))
                os.rename(os.path.join(img_dir, filename), os.path.join(img_dir, new_filename))

if __name__ == '__main__':
    main()
