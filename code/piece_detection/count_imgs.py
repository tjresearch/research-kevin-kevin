"""
utility script to count classes represented by images
(not used in final product)
"""

import sys, time
import cv2
import os
import numpy as np
from validate_imgs import get_human_label

"""
take folder of folder of folders of images as input
    labelled_squares
    |_outer_dir
        |_ img_dir
            |_ 0-x.jpg
            ...
        ...
    ...
(as produced by board/board_detection/piece_labelling.py)

then rename image files in place based on user input
"""
def main():
    stats = {}

    data_dir = sys.argv[1]
    print("searching {}...".format(data_dir))
    for file in os.listdir(data_dir):
        outer_dir = os.path.join(data_dir, file)
        if not os.path.isdir(outer_dir): continue
        print(outer_dir)
        for dir in os.listdir(outer_dir):
            img_dir = os.path.join(outer_dir, dir)
            for filename in os.listdir(img_dir):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    filepath = os.path.join(img_dir, filename)
                    # print("file: {}".format(filepath))
                    img = cv2.imread(filepath)

                    sq_num, label, ext = get_human_label(filename)

                    if label not in stats:
                        stats[label] = 0
                    stats[label] += 1

                    for word in label.split("_"):
                        if word not in stats:
                            stats[word] = 0
                        stats[word] += 1

    #add summaries
    totals = {}
    totals["pieces"] = stats["white"]+stats["black"]
    totals["squares"] = stats["empty"]+totals["pieces"]

    #display stats
    print("\n"+"-"*15)
    print("piece types:\n")
    sorted_stats = dict(sorted(stats.items()))
    for word, freq in sorted_stats.items():
        if " " in word:
            print("{}: {}".format(word, freq))
    print()

    for word, freq in sorted_stats.items():
        if " " not in word and word not in {"empty", "black", "white"}:
            print("{}: {}".format(word, freq))
    print("-"*15)


    print("totals:\n")
    print("{}: {}".format("black", stats["black"]))
    print("{}: {}".format("white", stats["white"]))
    print("{}: {}".format("empty", stats["empty"]))
    print()

    for word, freq in totals.items():
        print("{}: {}".format(word, freq))
    print("-"*15)

if __name__ == '__main__':
    main()
