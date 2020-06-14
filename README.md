# AutoPGN

AutoPGN is an automatic chess-move transcriber. It notates the moves played in a prerecorded chess game, in real-time, using computer vision and machine learning, then outputs a PGN file that can be copy-pasted into other chess software. It also updates a graphical representation of the board as it runs.

Paper: https://drive.google.com/file/d/12eamkGZ2owfkUtRWU2UreRRwwqyMmgmF/view?usp=sharing
Live Demo: https://youtu.be/WzbYgsyceso

## Code Overview:

```bash
code/
├── assets/
├── board_detection/
├── chess_logic/
├── download_models.sh
├── models/
├── piece_detection/
└── user_interface/
```

`assets/`
 - Input images for the image-based UI in `user_interface/`

`board_detection/` 
 - **main method: `board_detection/video_handler.py`**
 - Hough transform-based board detection
 - Board segmentation code
 - Lattice point CNN train script 

`chess_logic/`
 - PGN move transcription engine
 - Sample .pgn files

`download_models.sh`
 - Script to download lattice point CNN and piece detection CNN to `models/` from GCloud

`models/`
 - Directory to house model files 

`piece_detection/`
 - Piece detection CNN train scripts
 - Data augmentation utility scripts
 - Data collection utils 
 - Command-line video handler

`user_interface/`
 - Image-based UI (with output debug images saved to `user_interface/assets`)
 - Query script for graphical chessboard representation
 - Deprecated wifi-camera based input scripts (requires installation of [IPCamera](https://apps.apple.com/us/app/ipcamera-high-end-networkcam/id570912928)) 

## Getting Started

### Dependencies

AutoPGN is written in `python 3.6`. To run the main method, found in `board_detection/video_handler.py`, install the following packages:
 - numpy 1.17
 - opencv-python 4.1.0
 - tensorflow 2.0.0
 - scikit-learn 0.21.3

Some of the utility scripts in this repo are not part of the main video handling method. To run every script in this repo, install these additional packages:
 - matplotlib 3.1.1

### Running



## Sample Output:

Board detection:
![board detection](readme_images/line_detect_1018.png)
