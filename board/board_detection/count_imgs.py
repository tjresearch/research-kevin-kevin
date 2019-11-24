import sys, time
import cv2
import os
import numpy as np
from validate_imgs import get_human_label

"""
take folder of folders of images as input
    outer_dir
    |_ img_dir
        |_ 0-x.jpg
        ...
    ...
(as produced by board/board_detection/piece_labelling.py)

then tally piece/color types
"""
def main():
    stats = {
        "rook":0,
        "knight":0,
        "bishop":0,
        "king":0,
        "queen":0,
        "pawn":0,
        "empty":0,
        "white":0,
        "black":0,
    } #totals calculated later

    outer_dir = sys.argv[1]
    for dir in os.listdir(outer_dir):
        img_dir = os.path.join(outer_dir, dir)
        print("img_dir:",img_dir)
        for filename in os.listdir(img_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                filepath = os.path.join(img_dir, filename)
                # print("file: {}".format(filepath))
                img = cv2.imread(filepath)

                sq_num, label, ext = get_human_label(filename)
                for word in label.split(" "):
                    stats[word] += 1

    stats["total_pieces"] = stats["white"]+stats["black"]
    stats["total_squares"] = stats["empty"]+stats["total_pieces"]

    #display stats
    print("\n"+"-"*15)
    print("stats:\n")
    for word, freq in stats.items():
        print("{}: {}".format(word, freq))
    print("-"*15)

if __name__ == '__main__':
    main()
