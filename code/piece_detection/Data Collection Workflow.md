# Piece Data Gathering Workflow

## Video to Images

1. record chess game, make sure there is a pause between moves where hand leaves and reenters board

2. convert video to frames with video_handler.py:

   ```
   python video_handler.py [src_video] 0 [output_dir]
   ```

3. move generated frames and corner_cache.txt to appropriate folder in piece_detection/ for data_collection_ui.py

4. add appropriate filepaths to start of corner_cache.txt (CTRL-SHIFT-L is multiline on Atom)

5. skip "Preprocessing Images"

## Preprocessing Images (Mac)

1. get images of chess pieces on a board

2. airdrop from phone to laptop

3. convert from HEIC to .jpeg or .jpg with Preview

   ```bash
   File -> Export -> Options -> JPEG 
   pick directory to save to
   ```

4. delete HEIC images

5. resize JPEGs in Preview to 960x720

   ```bash
   Tools -> Adjust Size
   Width: 960
   Height: 720
   leave both checkboxes checked
   ```

6. move folder of JPEGs to code/piece_detection

## Labelling Pieces

1. after creating output dir, run this:

   ```bash
   python data_collection_ui.py input_dir output_dir [pgn_file] [white_on_left]
   ```

2. move img folders out of timestamped folders

3. format dataset (input dir is resulting dir from previous, output is "formatted_train_data")

   ```bash
   python format_train_data.py input_dir
   ```

4. ~~[OPTIONAL] run crop_train_data.py on resulting dir to crop images to fixed size~~

5. [OPTIONAL] run class_reducer.py on resulting dir to even out class counts

   ```
   python class_reducer.py [in_dir] [out_dir] [max_cls_size]
   
   ```

6. [OPTIONAL] merge new partial dataset with existing full dataset

   ```bash
   python merge_data.py [src] [dst]
   ```

7. split into train/test subsets

```bash
python train_valid_split.py [img_dir] [split_pct (decimal)]
```

8. ~~copy to snowy (see Working with Snowy.md)~~

