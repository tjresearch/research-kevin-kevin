import tkinter as tk
import cv2
import time
import os
import sys
import shutil
import random
from tkinter import filedialog, simpledialog
from tkinter.messagebox import showerror
from threading import Thread, Event

from PIL import Image, ImageTk

import query_diagram

sys.path.insert(1, "../board_detection")
import board_locator

sys.path.insert(2, "../piece_detection")
import piece_classifier

supported_image_formats = [".bmp", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".jpeg", ".jpg", ".jpe", ".jp2", ".tiff", ".tif", ".png"]
supported_video_formats = [".avi", ".flv", ".wmv", ".mov", ".mp4"]
TARGET_SIZE = (224, 112)

immortals = []

class Display(tk.Frame):
	def __init__(self, parent):
		self.image_dims = (960, 540)
		self.mode = ""

		self.live_cap = None
		self.live_video_thread = None
		self.live_video_stop = Event()
		self.live_video_play = Event()

		self.video_cap = None
		self.video_thread = None
		self.video_stop = Event()
		self.video_play = Event()

		self.display_thread = None
		self.cur_raw_image = None
		self.display_stop = Event()

		self.processing = False
		self.processing_thread = None

		self.board = None
		self.ortho_guesses = None

		self.intermediate_index = 0
		self.intermediate_image_order = ["raw.jpg",
										 "line_detection.jpg",
										 "line_linking.jpg",
										 "line_filtering.jpg",
										 "lattice_points.jpg",
										 "board_localization.jpg",
										 "board_segmentation.jpg",
										 "orthophoto_guesses.jpg",
										 "bounding_boxes.jpg",
										 "unsheared_sqrs",
										 "conf_intervals"]
		self.intermediate_image_dir = "./assets/intermediate_images"

		self.cur_animate_image = None
		self.animate_cycle = 50
		self.animate_index = 0

		self.reset_placeholders("./assets/intermediate_images/placeholders")

		tk.Frame.__init__(self, parent)

		self.button_frame = tk.Frame(self)

		# Initalize the input buttons
		self.side_button_frame = tk.Frame(self.button_frame)
		self.file_button = tk.Button(self.side_button_frame, command=lambda: self.update_display("image"), text="Image", height=3, width=16)
		self.video_button = tk.Button(self.side_button_frame, command=lambda: self.update_display("video"), text="Video", height=3, width=16)
		self.live_video_button = tk.Button(self.side_button_frame, command=lambda: self.update_display("live_video"), text="Live Video", height=3, width=16)

		self.file_button.pack(side=tk.TOP)
		self.video_button.pack(side=tk.TOP)
		self.live_video_button.pack(side=tk.TOP)
		self.side_button_frame.pack(side=tk.TOP)

		# Initialize the buttons to work with the backend
		self.bottom_button_frame = tk.Frame(self.button_frame)

		self.process_stop = Event()
		self.process_frame = tk.Frame(self.bottom_button_frame)
		self.process_button = tk.Button(self.process_frame, command=self.process, text="Process", height=3, width=16)

		self.prev_corners = None
		self.prev_raw_image = None

		self.use_prev_corners = tk.IntVar()
		self.prev_check = tk.Checkbutton(self.process_frame, text="Use Previous Corners", variable=self.use_prev_corners)

		self.process_button.pack(side=tk.TOP)
		# self.prev_check.pack(side=tk.BOTTOM)

		# Initialize the buttons that will tab through intermediate images
		self.nav_button_frame = tk.Frame(self.bottom_button_frame)
		self.back_button = tk.Button(self.nav_button_frame, command=self.back, text="Back", height=2, width=8)
		self.next_button = tk.Button(self.nav_button_frame, command=self.next, text="Next", height=2, width=8)

		self.back_button.pack(side=tk.LEFT)
		self.next_button.pack(side=tk.RIGHT)
		self.nav_button_frame.pack(side=tk.BOTTOM)

		self.process_frame.pack(side=tk.TOP)
		self.bottom_button_frame.pack(side=tk.BOTTOM, pady=20)

		self.button_frame.pack(side=tk.LEFT)

		# Intialize the display section
		self.display_frame = tk.Frame(self)

		self.image_label = tk.Label(self.display_frame)
		self.caption = tk.Label(self.display_frame)

		self.show_image("assets/intermediate_images/placeholders/raw.jpg")

		self.video_controls = tk.Frame(self.display_frame)
		self.prev_frame_button = tk.Button(self.video_controls, command=self.prev_raw_image, text="Prev Frame", height=2, width=8)
		self.pause_play_button = tk.Button(self.video_controls, command=self.pause_play, text="Pause/Play", height=2, width=8)
		self.next_frame_button = tk.Button(self.video_controls, command=self.next_frame, text="Next Frame", height=2, width=8)

		self.prev_frame_button.pack(side=tk.LEFT)
		self.pause_play_button.pack(side=tk.LEFT)
		self.next_frame_button.pack(side=tk.LEFT)

		self.side_display_frame = tk.Frame(self)

		self.status_frame = tk.Frame(self.side_display_frame)

		self.status_label = tk.Text(self.status_frame, width=40, height=12, font="Helvetica 16")
		self.status_label.tag_configure("bold", font="Helvetica 16 bold")
		self.status_label.insert("end", "Status:\n", "bold")
		self.status_label.configure(state="disabled")

		self.status_scroll = tk.Scrollbar(self.status_frame, orient="vertical", command=self.status_label.yview)
		self.status_label.configure(yscrollcommand=self.status_scroll.set)

		self.status_label.grid(row=0, column=0, rowspan=12, sticky="w")
		self.status_scroll.grid(row=0, column=1, rowspan=12, sticky="ns")

		# Intialize the FEN/PGN display
		self.diagram_id = None
		self.diagram_dim = 320
		self.selected_square = None

		self.diagram_canvas = tk.Canvas(self.side_display_frame, borderwidth=0, highlightthickness=0, width=self.diagram_dim, height=self.diagram_dim)
		self.diagram_canvas.bind("<Button-1>", self.diagram_mouse_callback)

		self.status_frame.pack(side=tk.TOP)
		self.diagram_canvas.pack(side=tk.BOTTOM)

		self.side_display_frame.pack(side=tk.RIGHT, padx=10, pady=10)

		self.image_label.pack(side=tk.TOP)
		self.caption.pack(side=tk.TOP)
		self.display_frame.pack(side=tk.RIGHT)

		self.pack(fill="both", expand="true", padx=4, pady=4)

		self.start_display()

		# Load models
		self.models_loaded = False
		self.lattice_point_model = None
		self.piece_model = None

		self.model_thread = Thread(target=self.load_models)
		self.model_thread.daemon = True
		self.model_thread.start()

	def reset_placeholders(self, placeholder_dir):
		for path in self.intermediate_image_order:
			if path.endswith(".jpg"):
				placeholder_image = path
			else:
				if os.path.isdir(os.path.join(self.intermediate_image_dir, path)):
					shutil.rmtree(os.path.join(self.intermediate_image_dir, path))
				placeholder_image = "{}.jpg".format(path)
			shutil.copy(os.path.join(placeholder_dir, placeholder_image), os.path.join(self.intermediate_image_dir, placeholder_image))

	def status_update(self, text):
		print(text)
		self.status_label.configure(state="normal")
		self.status_label.insert("end", text + "\n")
		self.status_label.configure(state="disabled")
		self.status_label.see("end")

	def load_models(self):
		model_dir = "../models"

		# Load board models
		self.status_update("Loading board model...")
		st_load_time = time.time()
		self.lattice_point_model = board_locator.load_model(os.path.join(model_dir, "lattice_points_model.json"),
													   os.path.join(model_dir, "lattice_points_model.h5"))
		self.status_update("> Loaded in {} s".format(time.time() - st_load_time))

		# Load piece models
		self.status_update("Loading piece model...")
		st_load_time = time.time()
		self.piece_model = piece_classifier.local_load_model(os.path.join(model_dir, "piece_detection_model.h5"))
		self.status_update("> Loaded in {} s".format(time.time() - st_load_time))

		self.models_loaded = True

	def update_display(self, new_mode):
		if not (self.mode == "live_video" and new_mode == "live_video") and not self.processing:
			old_mode = self.mode
			self.mode = new_mode

			if old_mode == "video":
				self.video_controls.pack_forget()
				self.video_stop.set()
			elif old_mode == "live_video":
				self.live_video_stop.set()
			elif old_mode == "image":
				if new_mode != "image":
					self.prev_check.pack_forget()

			if self.mode == "image":
				file = self.choose_file()
				if file is not None and any(file.endswith(ext) for ext in supported_image_formats):
					self.prev_check.pack(side=tk.BOTTOM)
					self.caption["text"] = file
					self.show_image(file)
			elif self.mode == "video":
				file = self.choose_file()
				if file is not None and any(file.endswith(ext) for ext in supported_video_formats):
					ret = self.start_video(file)
					if not ret:
						self.mode = old_mode
			elif self.mode == "live_video":
				ret = self.start_live_video()
				if not ret:
					self.mode = old_mode

			self.intermediate_index = 0

	def show_image(self, image_path):
		load = cv2.imread(image_path)
		self.cur_raw_image = load

	@staticmethod
	def choose_file():
		chosen_file = filedialog.askopenfilename(initialdir = "...", title="Select file")
		if chosen_file:
			return chosen_file
		return None

	def start_live_video(self):
		ip = simpledialog.askstring("IP", "IP Address:")

		if ip is None:
			return False

		url = "http://" + ip + "/live?type=some.mp4"
		self.live_cap = cv2.VideoCapture(url)
		if not self.live_cap.isOpened():
			showerror("Error", "Could not connect capture device at {}".format(ip))
			return False

		fps = self.live_cap.get(cv2.CAP_PROP_FPS)
		resolution = (int(self.live_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.live_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
		self.caption["text"] = "IP: {}\nFPS: {}\nResolution {}".format(ip, fps, resolution)

		self.live_video_play.set()
		self.live_video_thread = Thread(target=self.live_video_handler)
		self.live_video_thread.daemon = True
		self.live_video_thread.start()

		return True

	def live_video_handler(self):
		try:
			while not self.live_video_stop.is_set():
				self.live_video_play.wait()
				ret, frame = self.live_cap.read()
				if ret:
					self.cur_raw_image = frame
				else:
					break
		except RuntimeError:
			print("Caught a RuntimeError")
		self.live_cap.release()
		self.live_video_stop.clear()

	def start_display(self):
		self.display_thread = Thread(target=self.display_handler)
		self.display_thread.daemon = True
		self.display_thread.start()

	def display_handler(self):
		while not self.display_stop.is_set():
			if self.intermediate_index == 0:
				disp = Image.fromarray(cv2.cvtColor(cv2.resize(self.cur_raw_image, self.image_dims), cv2.COLOR_BGR2RGB))
			else:
				intermediate_path = self.intermediate_image_order[self.intermediate_index]
				if intermediate_path.endswith(".jpg"):
					image = Image.open(os.path.join(self.intermediate_image_dir, intermediate_path))
				else:
					cur_dir = os.path.join(self.intermediate_image_dir, intermediate_path)
					if os.path.isdir(cur_dir):
						if self.selected_square is None:
							if self.animate_index == 0:
								image_path = os.path.join(cur_dir, random.choice(os.listdir(cur_dir)))
								image = Image.open(image_path)
								self.cur_animate_image = image
							else:
								image = self.cur_animate_image
							self.animate_index += 1
							if self.animate_index == self.animate_cycle:
								self.animate_index = 0
						else:
							image_path = os.path.join(cur_dir, "sq_{}.jpg".format(self.selected_square))
							image = Image.open(image_path)
					else:
						image = Image.open(cur_dir + ".jpg")
				disp = image.resize(self.image_dims)
			render = ImageTk.PhotoImage(disp)
			self.image_label.configure(image=render)
			self.image_label.image = render

	def pause_play(self):
		if not self.processing:
			if self.video_play.is_set():
				self.video_play.clear()
			else:
				self.video_play.set()

	def prev_frame(self):
		if not self.processing:
			if not self.video_play.is_set():
				prev_pos = self.video_cap.get(cv2.CAP_PROP_POS_FRAMES) - 2
				if prev_pos >= 0:
					self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, prev_pos)
					self.disp_next_frame()

	def next_frame(self):
		if not self.processing:
			if not self.video_play.is_set():
				self.disp_next_frame()

	def disp_next_frame(self):
		ret, frame = self.video_cap.read()
		if ret:
			self.cur_raw_image = frame

	def start_video(self, video_path):
		self.video_controls.pack()
		self.video_cap = cv2.VideoCapture(video_path)
		if not self.video_cap.isOpened():
			showerror("Error", "Could not open video at {}".format(video_path))
			return False
		fps = self.video_cap.get(cv2.CAP_PROP_FPS)
		resolution = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
		self.caption["text"] = "File: {}\nFPS: {}\nResolution {}".format(video_path, fps, resolution)

		self.video_play.set()
		self.video_thread = Thread(target=self.video_handler, args=(fps,))
		self.video_thread.daemon = True
		self.video_thread.start()

		return True

	def video_handler(self, fps):
		try:
			while not self.video_stop.is_set():
				self.video_play.wait()
				ret, frame = self.video_cap.read()
				if ret:
					self.cur_raw_image = frame
					time.sleep(1 / fps)
		except RuntimeError:
			print("Caught a RuntimeError")
		self.video_cap.release()
		self.video_stop.clear()

	def process(self):
		if not self.processing:
			if self.models_loaded:
				if self.mode == "video" or self.mode == "live_video":
					self.process_button.configure(text="Stop Process")
					self.process_button["text"] = "Stop Process"
					self.process_stop.clear()
				self.processing_thread = Thread(target=self.process_handler)
				self.processing_thread.daemon = True
				self.processing_thread.start()
			else:
				showerror("Error", "Models are still loading, try again in a few seconds.")
		elif self.mode == "video" or self.mode == "live_video":
			self.process_button.configure(text="Process")
			self.process_button["text"] = "Process"
			self.process_stop.set()

	def process_handler(self):
		self.processing = True

		if self.mode == "video" or self.mode == "live_video":
			while not self.process_stop.is_set():
				self.process_single_frame()
		else:
			self.process_single_frame()

		self.processing = False

	def process_single_frame(self):
		self.status_update("Locating board...")
		st_locate_time = time.time()

		if self.mode == "image":
			if self.use_prev_corners.get():
				self.status_update("Using previous corners...")
				if self.prev_raw_image is None or self.prev_corners is None:
					self.status_update("No previous data found. Calculating from scratch...")
				lines, corners = board_locator.find_chessboard(self.cur_raw_image, self.lattice_point_model,
															   "./assets/intermediate_images",
															   (self.prev_raw_image, self.prev_corners))
			else:
				lines, corners = board_locator.find_chessboard(self.cur_raw_image, self.lattice_point_model,
															   "./assets/intermediate_images")
		else:
			lines, corners = board_locator.find_chessboard(self.cur_raw_image, self.lattice_point_model, prev=(self.prev_raw_image, self.prev_corners))

		self.prev_corners = [(corner[0], corner[1]) for corner in corners]
		self.prev_raw_image = self.cur_raw_image

		self.status_update("> Located board in {} s".format(time.time() - st_locate_time))

		self.status_update("Classifying pieces...")
		st_classify_time = time.time()
		if self.mode == "image":
			self.board, self.ortho_guesses = piece_classifier.classify_pieces(self.cur_raw_image, corners, self.piece_model, TARGET_SIZE,
												graphics_IO=("../assets", "./assets/intermediate_images"))
		else:
			self.board, self.ortho_guesses = piece_classifier.classify_pieces(self.cur_raw_image, corners, self.piece_model, TARGET_SIZE)
		self.status_update("> Classified pieces in {} s".format(time.time() - st_classify_time))

		board_string = "".join("".join(row) for row in self.board)

		self.status_update("Querying diagram...")
		st_query_time = time.time()
		diagram = query_diagram.diagram_from_board_string(board_string)
		self.status_update("> Queried diagram in {} s".format(time.time() - st_query_time))

		render = ImageTk.PhotoImage(diagram)
		if self.diagram_id is not None:
			self.diagram_canvas.delete(self.diagram_id)
		self.diagram_id = self.diagram_canvas.create_image((self.diagram_dim // 2, self.diagram_dim // 2), image=render)
		immortals.append(render)

		self.diagram_canvas.delete("square")
		self.selected_square = None

	def diagram_mouse_callback(self, event):
		square_size = self.diagram_dim // 8
		r = event.y // square_size
		c = event.x // square_size
		square_idx = c * 8 + (7 - r) # Account for rotation of diagram
		if self.selected_square == square_idx:
			self.diagram_canvas.delete("square")
			self.selected_square = None
		elif self.diagram_id is not None:
			if self.ortho_guesses[square_idx]:
				x1 = c * square_size
				y1 = r * square_size
				x2 = x1 + square_size
				y2 = y1 + square_size
				self.diagram_canvas.delete("square")
				self.selected_square = square_idx
				self.diagram_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=3, tags="square")
			else:
				self.status_update("Square is not in orthophoto; pick a different square!")

	def back(self):
		if self.intermediate_index > 0:
			self.intermediate_index -= 1

	def next(self):
		if self.intermediate_index < len(self.intermediate_image_order) - 1:
			self.intermediate_index += 1


root = tk.Tk()
root.title("AutoPGN")
disp = Display(root)
root.mainloop()
