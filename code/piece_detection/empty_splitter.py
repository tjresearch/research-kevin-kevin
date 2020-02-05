"""
script to reduce size of empty class, randomly
"""

import sys
import os
import shutil
import random

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    red_pct = float(sys.argv[3])
    print(sys.argv)
    if len(sys.argv) < 3:
        exit("Usage: python empty_splitter.py [in_dir] [out_dir] [reduction_pct]")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        exit("{} exists".format(output_dir))

    for img_file in os.listdir(input_dir):
        if random.random() < red_pct:
            src = os.path.join(input_dir, img_file)
            dst = os.path.join(output_dir, img_file)
            print(src, " -> ", dst)
            shutil.copyfile(src, dst)

if __name__ == '__main__':
    main()
