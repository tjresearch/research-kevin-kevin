import tkinter as tk
import sys
import cv2
import os
import time
from tkinter.messagebox import askyesno
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np

from square_splitter import split_chessboard, order_points
from piece_classifier import local_load_model, pred_squares
sys.path.insert(1, "../board_detection")
import board_locator, board_segmentation

immortals = {}
notation = {"white_pawn":"P", "white_knight":"N", "white_bishop":"B", "white_rook":"R", "white_queen":"Q", "white_king":"K",
			"black_pawn":"p", "black_knight":"n", "black_bishop":"b", "black_rook":"r", "black_queen":"q", "black_king":"k"}
rev_notation = {v:k for k, v in notation.items()}

hotkeys = {"<space>":"toggle", "x":"clear", "q":"pawn", "a":"knight", "s":"bishop", "d":"rook", "w":"queen", "e":"king"}

print("loading piece detection model...")
piece_detection_model = local_load_model("../models/piece_detection_model.h5")
TARGET_SIZE = (224, 112)
print("piece detection model loaded")

class DataCollectionDisp(tk.Frame):
	def __init__(self, parent, img, squares, indices, save_dir):

		self.parent = parent
		tk.Frame.__init__(self, parent)

		self.squares = squares
		self.indices = indices
		self.save_dir = save_dir

		self.colors = ["#FFE1BB", "#EBD297"]

		self.square_size = 80

		self.selected_piece = ""

		self.top_frame = tk.Frame(self)

		self.board_canvas = tk.Canvas(self.top_frame, borderwidth=0, highlightthickness=0, width=self.square_size * 8, height=self.square_size * 8)
		self.board_ids = []
		self.board = []

		pred_board = pred_squares(TARGET_SIZE, piece_detection_model, squares, indices) #get nnet preds
		#rotate board for std display (white on bottom)
		#converting to numpy and back takes 0.0 s (rounded to 3 digits)
		pred_board = np.asarray(pred_board)
		pred_board = np.resize(pred_board, (8,8))
		pred_board = np.rot90(pred_board, 3)
		self.preds = [[None for j in range(8)] for i in range(8)]
		for i in range(8):
			for j in range(8):
				self.preds[i][j] = pred_board[i][j]

		self.init_board()

		self.board_canvas.pack(side=tk.LEFT)
		self.board_canvas.bind("<Button-1>", self.mouse_callback)

		for key in hotkeys:
			self.board_canvas.bind(key, lambda event, i=hotkeys[key]: self.hotkey_handler(event, i))

		self.board_canvas.focus_set()

		self.image_label = tk.Label(self.top_frame)

		# disp = Image.fromarray(cv2.cvtColor(cv2.resize(img, None, fx=0.75, fy=0.75), cv2.COLOR_BGR2RGB))
		disp = Image.fromarray(cv2.cvtColor(cv2.resize(img, None, fx=0.6, fy=0.6), cv2.COLOR_BGR2RGB))
		render = ImageTk.PhotoImage(disp)
		self.image_label.configure(image=render)
		self.image_label.image = render

		self.image_label.pack(side=tk.RIGHT)

		self.middle_frame = tk.Frame(self.top_frame)

		self.button_frame = tk.Frame(self.middle_frame)

		# self.selected_piece_label = tk.Label(self.middle_frame, text="Selected: clear")
		# self.selected_piece_label.pack(side=tk.TOP)

		self.white_frame = tk.Frame(self.button_frame)
		self.black_frame = tk.Frame(self.button_frame)

		pieces = ["pawn", "knight", "bishop", "rook", "queen", "king"]

		for piece in pieces:
			img = Image.open("../assets/piece_images/white_{}.png".format(piece))
			img = img.resize((self.square_size, self.square_size))
			render = ImageTk.PhotoImage(img)
			button = tk.Button(self.white_frame, command=lambda i="white_{}".format(piece): self.piece_button_callback(i), image=render, highlightbackground="black")
			button.image = render
			button.pack(side=tk.TOP)

			img = Image.open("../assets/piece_images/black_{}.png".format(piece))
			img = img.resize((self.square_size, self.square_size))
			render = ImageTk.PhotoImage(img)
			button = tk.Button(self.black_frame, command=lambda i="black_{}".format(piece): self.piece_button_callback(i), image=render, highlightbackground="black")
			button.image = render
			button.pack(side=tk.TOP)

		button = tk.Button(self.button_frame, width=10, height=5, command=lambda : self.piece_button_callback(""), text="Clear square", bg="white")
		button.pack(side=tk.BOTTOM)

		self.white_frame.pack(side=tk.LEFT)
		self.black_frame.pack(side=tk.RIGHT)

		self.button_frame.pack(side=tk.BOTTOM)

		self.middle_frame.pack(side=tk.RIGHT, padx=10)

		self.top_frame.pack(side=tk.TOP)

		self.save_button = tk.Button(self, width=30, height=5, command=self.save, text="Save")
		self.save_button.pack(side=tk.BOTTOM)

		self.pack(fill="both", expand="true")

	def init_board(self):
		# temp
		for temp in self.preds:
			print(temp)

		for r in range(8):
			row = []
			row_ids = []
			for c in range(8):
				x1 = c * self.square_size
				y1 = r * self.square_size
				x2 = x1 + self.square_size
				y2 = y1 + self.square_size
				row.append("x")
				row_ids.append(self.board_canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=(self.colors[0] if (r + c) % 2 else self.colors[1])))
			self.board.append(row)
			self.board_ids.append(row_ids)

		# start board with nnet preds
		for r in range(8):
			for c in range(8):
				if self.preds[r][c] == "-": continue
				x = c * self.square_size + self.square_size // 2
				y = r * self.square_size + self.square_size // 2
				self.selected_piece = rev_notation[self.preds[r][c]]
				img = Image.open("../assets/piece_images/{}.png".format(self.selected_piece))
				img = img.resize((self.square_size, self.square_size))
				render = ImageTk.PhotoImage(img)
				self.board_canvas.delete(self.board_ids[r][c])
				self.board[r][c] = notation[self.selected_piece]
				self.board_ids[r][c] = self.board_canvas.create_image((x, y), image=render)
				immortals[self.board_ids[r][c]] = render

	def piece_button_callback(self, piece):
		self.selected_piece = piece
		new_text = "Selected: {}".format(piece if piece else "clear")
		# self.selected_piece_label.configure(text=new_text)
		# self.selected_piece_label.text = new_text

	def hotkey_handler(self, event, command):
		if command == "toggle":
			cur_color = self.selected_piece[:self.selected_piece.find("_")]
			new_color = "black" if cur_color == "white" else "white"
			self.selected_piece = new_color + self.selected_piece[self.selected_piece.find("_"):]
		elif command == "clear":
			self.selected_piece = ""
		else:
			if self.selected_piece:
				self.selected_piece = self.selected_piece[:self.selected_piece.find("_") + 1] + command
			else:
				self.selected_piece = "white_{}".format(command)
		new_text = "Selected: {}".format(self.selected_piece if self.selected_piece else "clear")
		# self.selected_piece_label.configure(text=new_text)
		# self.selected_piece_label.text = new_text

	def mouse_callback(self, event):
		global immortals
		r = event.y // self.square_size
		c = event.x // self.square_size
		x = c * self.square_size + self.square_size // 2
		y = r * self.square_size + self.square_size // 2
		if self.selected_piece:
			img = Image.open("../assets/piece_images/{}.png".format(self.selected_piece))
			img = img.resize((self.square_size, self.square_size))
			render = ImageTk.PhotoImage(img)
			self.board_canvas.delete(self.board_ids[r][c])
			self.board[r][c] = notation[self.selected_piece]
			self.board_ids[r][c] = self.board_canvas.create_image((x, y), image=render)
			immortals[self.board_ids[r][c]] = render
		else:
			x1 = c * self.square_size
			y1 = r * self.square_size
			x2 = x1 + self.square_size
			y2 = y1 + self.square_size
			self.board_canvas.delete(self.board_ids[r][c])
			self.board[r][c] = "x"
			self.board_ids[r][c] = self.board_canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=(self.colors[0] if (r + c) % 2 else self.colors[1]))

	def save(self):
		result = askyesno("Save", "Are you sure you want to save?")
		if result:
			for r in range(8):
				for c in range(8):
					indx = r * 8 + c
					if indx in self.indices:
						i = self.indices.index(indx)
						img = self.squares[i]

						filename = "{}-{}.jpg".format(indx, self.board[r][c])
						full_path = os.path.join(self.save_dir, filename)
						cv2.imwrite(full_path, img)
			self.parent.destroy()

corners = []
def mark_point(event, x, y, flags, params):
	global corners
	if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
		print("Marked: {}, {}".format(x, y))
		corners.append((x, y))

def find_board(img, file, board_corners):
	cv2.namedWindow("full_image")
	global corners

	cv2.imshow("full_image", img)

	disp = img.copy()

	corners = board_corners[file]

	for corner in corners:
		cv2.circle(disp, (int(corner[0]), int(corner[1])), 3, (255, 0, 0), 2)
	cv2.imshow("full_image", disp)

	c = chr(cv2.waitKey())
	if c != " ":
		#manual override on corners from board_locator
		#reselect corners of board to segment
		corners = []
		cv2.imshow("full_image", img)

		print("ESC to quit")
		while True:
			cv2.setMouseCallback("full_image", mark_point)

			while True:
				cv2.imshow("full_image", img)
				print("pick four corners, space to finish, any other to redo")

				c = chr(cv2.waitKey())
				if c == " ":
					break
				elif c == "\x1b":
					exit("escaped")
				else:
					corners = []
					print("corners cleared")

			disp = img.copy()
			corners = order_points(corners)

			for corner in corners:
				cv2.circle(disp, (int(corner[0]), int(corner[1])), 3, (255, 0, 0), 2)

			cv2.imshow("full_image", disp)
			print("space to confirm board, any other to redo")

			c = chr(cv2.waitKey())
			if c == " ":
				break
			elif c == "\x1b":
				exit("escaped")
			else:
				corners = []
				print("corners cleared")

	return corners

def label_subimgs(img, squares, indices, save_dir, root):
	window = DataCollectionDisp(root, img, squares, indices, save_dir)
	root.mainloop()

def save_squares(file, outer_dir, lattice_point_model, root, board_corners):
	#setup file IO
	save_dir = os.path.join(outer_dir, file[file.rfind("/")+1:file.rfind(".")])
	os.mkdir(save_dir)
	print("output dir: {}".format(save_dir))
	img = cv2.imread(file)

	#find corners of board
	corners = find_board(img, file, board_corners)

	#take corners, split image into subimgs of viable squares & their indices
	squares, indices = split_chessboard(img, corners)
	#label squares with pieces, save
	label_subimgs(img, squares, indices, save_dir, root)

def locate_corners(files, cache_file_path, lattice_point_model):

	cache = {}
	if os.path.exists(cache_file_path):
		with open(cache_file_path, "r") as infile:
			for line in infile.readlines():
				line = line.strip().split(" - ")
				cache[line[0]] = eval(line[1])
		cache_file = open(cache_file_path, "a")
	else:
		cache_file = open(cache_file_path, "w")

	out_dict = {}
	for i in range(len(files)):
		file = files[i]
		print("Locating corners for file {}/{} - {}".format(i + 1, len(files), file))
		if file in cache:
			out_dict[file] = cache[file]
			print("File in cache - skipping...")
		else:
			st_time = time.time()
			img = cv2.imread(file)
			lines, corners = board_locator.find_chessboard(img, lattice_point_model)
			out_dict[file] = corners
			cache_file.write("{} - {}\n".format(file, str(corners)))
			cache_file.flush()
			print("Located in {} s".format(time.time() - st_time))

	cache_file.close()
	return out_dict

def main():
	print("Loading board model...")
	model_dir = "../models"
	st_load_time = time.time()
	lattice_point_model = board_locator.load_model(os.path.join(model_dir, "lattice_points_model.json"),
												   os.path.join(model_dir, "lattice_points_model.h5"))
	print("Loaded in {} s".format(time.time() - st_load_time))

	global corners
	img_dir_path = sys.argv[1]

	# make dir of current time for subimgs
	now = datetime.now()
	today_dir = now.strftime("%Y%m%d%H%M%S")
	head = sys.argv[2]  # output dir
	save_dir = os.path.join(head, today_dir)
	os.mkdir(save_dir)
	print("save dir: {}".format(save_dir))

	ct = 0

	files = []
	for file in os.listdir(img_dir_path):
		if file.startswith("*"):
			continue
		if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
			filepath = os.path.join(img_dir_path, file)
			files.append(filepath)

	board_corners = locate_corners(files, "cache.txt", lattice_point_model)

	# save squares of each file
	for file in files:
		ct += 1
		print("img {}/{}".format(ct, len(files)))
		print("file: {}".format(file))

		root = tk.Tk()
		root.title("Data Collection")
		save_squares(file, save_dir, lattice_point_model, root, board_corners)
		corners = []  # clear for next board
		os.rename(file, os.path.join(img_dir_path, "*{}".format(os.path.basename(file))))  # mark as done

	# mark whole dir as done
	last_dir_i = img_dir_path[0:len(img_dir_path) - 1].rfind("/")
	os.rename(img_dir_path,
			  os.path.join(img_dir_path[:last_dir_i], "*{}".format(img_dir_path[last_dir_i + 1:])))  # mark as done

if __name__ == "__main__":
	main()
