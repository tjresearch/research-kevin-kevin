"""
DEPRECATED
utility that crops width of formatted train data
(run after calling format_train_data.py)
Usage: python crop_train_data.py input_dir_of_sqr_imgs
"""
import sys
import os
from PIL import Image

def main():
    #find min size to crop to
    min_size = []
    input_dir = sys.argv[1]
    for dirname in os.listdir(input_dir):
        if dirname == ".DS_Store": continue
        img_dir = os.path.join(input_dir, dirname)
        for filename in os.listdir(img_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                src_filepath = os.path.join(img_dir, filename)
                img = Image.open(src_filepath)

                pix_ct = img.size[0]*img.size[1]
                if len(min_size) == 0:
                    min_size = [pix_ct, img.size]
                if pix_ct < min_size[0]:
                    min_size[0] = pix_ct
                    min_size[1] = img.size
    print("Cropping to min size:", min_size)

    #then crop all imgs to that size (preserve bottom)
    dst_w, dst_h = min_size[1]
    for dirname in os.listdir(input_dir):
        if dirname == ".DS_Store": continue
        img_dir = os.path.join(input_dir, dirname)
        for filename in os.listdir(img_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                src_filepath = os.path.join(img_dir, filename)
                src = Image.open(src_filepath)

                src_w, src_h = src.size
                left = int((src_w-dst_w)/2)
                right = left+dst_w
                top = 0
                bottom = src_h

                dst = src.crop((left, top, right, bottom))
                if dst.size[0] != dst_w:
                    print(dst.size)
                    print(min_size[1])
                    print(src_filepath)
                    exit("size mismatch")

                dst.save(src_filepath) #overwrite src file

if __name__ == '__main__':
    main()
