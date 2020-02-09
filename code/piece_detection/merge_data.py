"""
script to merge data (by class) after format_train_data.py has been called
"""

import sys, os

if len(sys.argv) < 2:
    exit("Usage: python merge_data.py SRC_DIR DST_DIR")

SRC_DIR = sys.argv[1]
DST_DIR = sys.argv[2]

for class_dir in os.listdir(SRC_DIR):
    in_dir = os.path.join(SRC_DIR, class_dir)
    for img_file in os.listdir(in_dir):
        img_path = os.path.join(in_dir, img_file)
        out_dir = os.path.join(DST_DIR, class_dir, img_file)
        print("{} -> {}".format(img_path, out_dir))
        os.rename(img_path, out_dir)
