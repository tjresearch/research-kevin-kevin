"""
splits class separated data into train/validation subsets
"""

import sys
import os
import shutil
import random

def main():
    if len(sys.argv) < 2:
        exit("Usage: python train_valid_split.py input_dir validation_pct")
    input_dir = sys.argv[1]
    valid_pct = float(sys.argv[2])

    #make blank validation dir
    valid_dir = "split_valid_data"
    if os.path.exists(valid_dir):
        shutil.rmtree(valid_dir)
    os.mkdir(valid_dir)

    #copy full input dataset to new folder
    #(will be train after file moves)
    train_dir = "split_train_data"
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    shutil.copytree(input_dir, train_dir)

    for classname in os.listdir(train_dir):
        if classname.startswith("."): continue
        #make new class in validation set if needed
        valid_class_dir = os.path.join(valid_dir, classname)
        if classname not in os.listdir(valid_dir):
            os.mkdir(valid_class_dir)

        train_class_dir = os.path.join(train_dir, classname)
        img_list = os.listdir(train_class_dir)
        train_num = int(valid_pct*len(img_list))

        print("{}: {}".format(classname, train_num))
        valid_imgs = random.sample(img_list, train_num)

        for img in valid_imgs:
            src = os.path.join(train_class_dir, img)
            dst = os.path.join(valid_class_dir, img)
            print(src, " -> ", dst)
            shutil.move(src, dst)
        print()

if __name__ == '__main__':
    main()
