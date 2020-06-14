"""
turn output of data_collection_ui.py into Keras-compatible class dirs
will rename sqr imgs to [square #]_[original_imgname]
Usage: python format_train_data.py dcu_output_dir
"""

import sys
import os
import shutil

"""
return SAN of input piece and color
"""
def piece_label_handler(window_name):
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

	cv2.destroyWindow(window_name)

	#convert input color+piece to SAN
	if color == "w":
		piece = piece.upper()
	if not piece: #blank sq
		piece = "x"

	return piece

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
