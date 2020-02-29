import tkinter as tk
from PIL import Image, ImageTk

class DataCollectionDisp(tk.Frame):
	def __init__(self, parent):

		tk.Frame.__init__(self, parent)

		self.square_size = 100

		self.board_frame = tk.Frame(self)
		self.board = []

		self.init_board()

		self.board_frame.pack(side=tk.LEFT)

		pieces = ["pawn", "knight", "bishop", "rook", "queen", "king"]

		self.button_frame = tk.Frame(self)

		self.white_frame = tk.Frame(self.button_frame)
		self.black_frame = tk.Frame(self.button_frame)

		for piece in pieces:
			white_img = Image.open("piece_images/white_{}.png".format(piece))
			white_img = white_img.resize((self.square_size, self.square_size))
			white_render = ImageTk.PhotoImage(white_img)
			white_button = tk.Button(self.white_frame, command=lambda : self.piece_button_callback("white_{}".format(piece)), image=white_render, highlightbackground="black")
			white_button.image = white_render

			black_img = Image.open("piece_images/black_{}.png".format(piece))
			black_img = black_img.resize((self.square_size, self.square_size))
			black_render = ImageTk.PhotoImage(black_img)
			black_button = tk.Button(self.black_frame, command=lambda: self.piece_button_callback("black_{}".format(piece)), image=black_render, highlightbackground="black")
			black_button.image = black_render

			white_button.pack(side=tk.TOP)
			black_button.pack(side=tk.TOP)

		self.white_frame.pack(side=tk.LEFT)
		self.black_frame.pack(side=tk.RIGHT)

		self.button_frame.pack(side=tk.RIGHT, padx=10)

		self.pack(fill="both", expand="true")

	def init_board(self):
		color1 = "#FFE1BB"
		color2 = "#EBD297"
		for r in range(8):
			row = []
			for c in range(8):
				temp_frame = tk.Frame(self.board_frame, width=self.square_size, height=self.square_size)
				temp_frame.pack_propagate(0)
				temp = tk.Label(temp_frame, width=10, height=10, bg=(color1 if (r + c) % 2 else color2))
				temp.pack(fill=tk.BOTH)
				temp_frame.grid(row=r, column=c)

	def piece_button_callback(self, piece):
		print(piece)

root = tk.Tk()
root.title("Data Collection")
disp = DataCollectionDisp(root)
root.mainloop()