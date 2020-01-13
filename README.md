# AutoPGN

AutoPGN is an automatic chess-move transcriber. It locates a chessboard on an image, segments it into individual squares, then recognizes the piece (or lack thereof) on each square with computer vision and machine learning.

## Code Overview:

```bash
code/
├── board_detection
├── piece_detection
├── preprocessing
└── user_interface
```

`board_detection`: Hough line-based board detection, lattice point CNN

`piece_detection`: piece labelling utilities, piece detection CNN

`preprocessing`: demos that broadcast and record ipCamera input

`user_interface`: PGN reader & writer, sample .pgn files

## Requirements:

 - python 3.6
 - matplotlib 3.1.1
 - numpy 1.17
 - opencv-python 4.1.0
 - scikit-learn 0.21.3
 - tensorflow 2.0.0

## Installation Instructions:

1. Install required packages with pip (see above).
2. Download [ipCamera app](https://apps.apple.com/us/app/ipcamera-high-end-networkcam/id570912928) for iPhone.
3. Clone this repo.

## Run Instructions:

1. Open ipCamera on iPhone.
2. Note URL on app.

Currently, there's no working product, but putting the url from step 2 as a command-line arg of  `board_detection/live_line_detection.py` shows the board detection working live, when linked to the ipCamera app.

## Sample Output:

Board detection:
![board detection](readme_images/line_detect_1018.png)
