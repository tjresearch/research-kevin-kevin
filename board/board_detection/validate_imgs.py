import sys, time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from matplotlib.pyplot import imread, imshow, subplots, show
import cv2
import os
import numpy as np

"""
takes folder of folders of images
(as produced by board/board_detection/piece_labelling.py)
renames image files in place
"""
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

            dash = filename.index('-')
            dot = filename.index('.')
            sq_num = filename[:dash]
            piece = filename[dash+1:dot]
            ext = filename[dot+1:]
            print("label:", end=" ")
            if piece == "x":
                print("blank square")
            else:
                printout = ""
                if piece.islower():
                    color = "black"
                else:
                    color = "white"

                printout += color+" "
                p = piece.lower()
                if p == "r":
                    printout += "rook"
                elif p == "n":
                    printout += "knight"
                elif p == "b":
                    printout += "bishop"
                elif p == "k":
                    printout += "king"
                elif p == "q":
                    printout += "queen"
                elif p == "p":
                    printout += "pawn"
                print(printout)

            print("\nspace if label correct, ESC to quit, any other key to relabel")
            c = chr(cv2.waitKey())
            if c == "\x1b":
                exit("escaped")
            elif c == " ":
                cv2.destroyWindow(filename)
                continue

            print("relabelling", filename)
            while True:
                color = "?"
                print("select color")
                c = chr(cv2.waitKey())
                if c in {"w", "b"}:
                	color = c
                elif c in {" ", "e"}:
                	color = "e"
                elif c == "\x1b":
                	exit("escaped")

                print("select piece")
                piece = ""
                if color != "e":
                	piece = "?"
                	c = chr(cv2.waitKey())
                	if c in {"p", "q", "r", "n", "k", "b", "e"}:
                		piece = c
                	elif c == "\x1b":
                		exit("escaped")

                print("space to confirm, any other to redo")
                if piece:
                	print("color: {}\npiece: {}".format(color, piece))
                else:
                	print("blank square")

                if chr(cv2.waitKey()) == " ":
                	break
                elif c == "\x1b":
                	exit("escaped")
                else:
                	print("redo")

            cv2.destroyWindow(filename)
            if color == "w":
                piece = piece.upper()

            if not piece: #blank sq
                piece = "x"

            new_filename = "{}-{}.{}".format(sq_num, piece, ext)

            print("renaming {} to {}".format(filename, new_filename))
            os.rename(os.path.join(img_dir, filename), os.path.join(img_dir, new_filename))
