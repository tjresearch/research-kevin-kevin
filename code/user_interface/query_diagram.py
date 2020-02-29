import requests
import shutil
from PIL import Image
from io import BytesIO

def fen_to_board_string(fen):
	board = "".join(fen.split(" ")[0].split("/"))
	i = 0
	while i < len(board):
		if board[i].isnumeric():
			num = int(board[i])
			board = board[:i] + "-" * num + board[i + 1:]
			i += num
		else:
			i += 1
	return board

def diagram_from_fen(fen, outfile):
	board_string = fen_to_board_string(fen)
	url = "http://www.jinchess.com/chessboard/?p={}&ps=alpha-flat".format(board_string)
	response = requests.get(url, stream=True)
	with open(outfile, "wb") as out:
		shutil.copyfileobj(response.raw, out)
	del response

def diagram_from_board_string(board_string):
	url = "http://www.jinchess.com/chessboard/?p={}&ps=alpha-flat".format(board_string)
	response = requests.get(url, stream=True)
	img = Image.open(BytesIO(response.content))
	del response
	return img

if __name__ == "__main__":
	fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
	diagram_from_fen(fen, "image.png")