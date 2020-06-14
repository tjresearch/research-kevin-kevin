"""
script to reduce size of larger classes (post format_train_data.py)
input_dir - output dir of format_train_data.py
output_dir - new reduced dir, cannot exist beforehand
red_num - desired max # of imgs in class
"""

import sys
import os
import shutil
import random

def main():
    if len(sys.argv) < 3:
        exit("Usage: python class_reducer.py in_dir out_dir max_cls_size")

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    red_num = int(sys.argv[3])
    print(sys.argv)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        exit("{} exists".format(output_dir))

    for cls_name in os.listdir(input_dir):
        if cls_name.startswith("."): continue
        cls_dir = os.path.join(input_dir, cls_name)

        out_cls_dir = os.path.join(output_dir, cls_name)
        os.mkdir(out_cls_dir)

        transfer_set = os.listdir(cls_dir)
        if len(transfer_set) > red_num:
            print(len(transfer_set))
            print(red_num)
            transfer_set = random.sample(transfer_set, red_num)
            print(len(transfer_set))

        for img_file in transfer_set:
            src = os.path.join(cls_dir, img_file)
            dst = os.path.join(out_cls_dir, img_file)
            # print(src, " -> ", dst)
            shutil.copyfile(src, dst)

if __name__ == '__main__':
    main()
