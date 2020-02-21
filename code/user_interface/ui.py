import tkinter as tk
import cv2
from tkinter import filedialog, simpledialog
from tkinter.messagebox import showerror
from threading import Thread, Event

from PIL import Image, ImageTk

class Display(tk.Frame):
	def __init__(self, parent):
		self.image_dims = (960, 540)
		self.mode = ""
		self.live_video_thread = None
		self.live_video_stop = Event()
		self.cap = None

		self.intermediate_image_order = ["raw",
										 "line_detection",
										 "line_linking",
										 "lattice_points",
										 "board_localization",
										 "board_segmentation",
										 "piece_classification"]
		self.intermediate_image_dir = "./intermediate_images"

		tk.Frame.__init__(self, parent)

		self.button_frame = tk.Frame(self)

		# Initalize the input buttons
		self.side_button_frame = tk.Frame(self.button_frame)
		self.file_button = tk.Button(self.side_button_frame, command=lambda: self.update_display("image"), text="File", height=3, width=16)
		self.live_video_button = tk.Button(self.side_button_frame, command=lambda: self.update_display("live_video"), text="Live Video", height=3, width=16)

		self.file_button.pack(side=tk.TOP)
		self.live_video_button.pack(side=tk.TOP)
		self.side_button_frame.pack(side=tk.TOP)

		# Initialize the buttons to work with the backend
		self.bottom_button_frame = tk.Frame(self.button_frame)
		self.process_button = tk.Button(self.bottom_button_frame, text="Process", height=3, width=16)

		# Initialize the buttons that will tab through intermediate images
		self.nav_button_frame = tk.Frame(self.bottom_button_frame)
		self.back_button = tk.Button(self.nav_button_frame, text="Back", height=2, width=8)
		self.next_button = tk.Button(self.nav_button_frame, text="Next", height=2, width=8)

		self.back_button.pack(side=tk.LEFT)
		self.next_button.pack(side=tk.RIGHT)
		self.nav_button_frame.pack(side=tk.BOTTOM)

		self.process_button.pack(side=tk.TOP)
		self.bottom_button_frame.pack(side=tk.BOTTOM, pady=20)

		self.button_frame.pack(side=tk.LEFT)

		# Intialize the display section
		self.display_frame = tk.Frame(self)

		self.image_label = tk.Label(self.display_frame)
		self.caption = tk.Label(self.display_frame)

		self.show_image("assets/intermediate_images/raw.png")

		# Intialize the FEN/PGN display
		self.fen_pgn_label = tk.Label(self.display_frame, height=25, width=40, padx=20, pady=10, anchor="nw")
		self.fen_pgn_label.configure(font=("Helvetica", 20))

		self.fen_pgn_label.pack(side=tk.RIGHT)

		self.image_label.pack(side=tk.TOP)
		self.caption.pack(side=tk.TOP)
		self.display_frame.pack(side=tk.RIGHT)

		self.pack(fill="both", expand="true", padx=4, pady=4)

	def update_display(self, new_mode):
		if not (self.mode == "live_video" and new_mode == "live_video"):
			old_mode = self.mode
			self.mode = new_mode

			if old_mode == "live_video":
				self.live_video_stop.set()
				self.cap.release()

			if self.mode == "image":
				self.fen_pgn_label["text"] = "FEN:\n"
				self.choose_file()
			elif self.mode == "live_video":
				self.fen_pgn_label["text"] = "PGN:\n"
				self.start_live_video()

	def show_image(self, image_path):
		load = Image.open(image_path)
		load = load.resize(self.image_dims)
		render = ImageTk.PhotoImage(load)
		self.image_label.configure(image=render)
		self.image_label.image = render

	def choose_file(self):
		chosen_file = filedialog.askopenfilename(initialdir = "...", title="Select file")
		if chosen_file:
			self.caption["text"] = chosen_file
			self.show_image(chosen_file)

	def start_live_video(self):
		ip = simpledialog.askstring("IP", "IP Address:")

		url = "http://" + ip + "/live?type=some.mp4"
		self.cap = cv2.VideoCapture(url)
		if not self.cap.isOpened():
			showerror("Error", "Could not connect capture device at {}".format(ip))
			return
		fps = self.cap.get(cv2.CAP_PROP_FPS)
		resolution = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
		self.caption["text"] = "IP: {}\nFPS: {}\nResolution {}".format(ip, fps, resolution)

		self.live_video_thread = Thread(target=self.live_video_handler)
		self.live_video_thread.daemon = True
		self.live_video_thread.start()

	def live_video_handler(self):
		try:
			while not self.live_video_stop.is_set():
				ret, frame = self.cap.read()
				if ret:
					frame = cv2.resize(frame, self.image_dims)
					disp = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
					render = ImageTk.PhotoImage(disp)
					self.image_label.configure(image=render)
					self.image_label.image = render
				else:
					break
		except RuntimeError:
			print("Caught a RuntimeError")


	def process(self):
		pass

	def back(self):
		pass

	def forward(self):
		pass


root = tk.Tk()
root.title("AutoPGN")
disp = Display(root)
root.mainloop()