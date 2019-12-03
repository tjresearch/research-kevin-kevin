"""
this script makes validate_imgs unnecessary
if mislabelled img found, just move its class dir
"""
import sys
import os
from validate_imgs import get_human_label
import shutil

"""
first, copy labelled_squares
then, remove high_split/low_split info

next, run this to:
    rename imgs to [square #]_[original_imgname]
    sort into folders by class (black_rook, etc)

then delete "?" directory
"""
def main():
    input_dir = sys.argv[1]
    output_dir = "formatted_train_data"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        exit("dir exists")

    seen_dirs = {*os.listdir(output_dir)}

    for dirname in os.listdir(input_dir):
        if dirname == ".DS_Store": continue
        img_dir = os.path.join(input_dir, dirname)
        for filename in os.listdir(img_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                src_filepath = os.path.join(img_dir, filename)
                sq_num, label, ext = get_human_label(filename)

                class_dir = os.path.join(output_dir, label)
                if label not in seen_dirs:
                    seen_dirs.add(label)
                    os.mkdir(class_dir)
                new_filename = "{}_{}.{}".format(sq_num, dirname, ext)
                dst_filepath = os.path.join(class_dir, new_filename)
                print(src_filepath, " -> ", dst_filepath)
                shutil.copyfile(src_filepath, dst_filepath)

if __name__ == '__main__':
    main()
