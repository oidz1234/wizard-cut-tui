# WizardCut TUI

A simple terminal-based video editor that lets you cut parts of videos by deleting text.

## Installation


1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure FFmpeg is installed on your system:

## Usage

### Basic usage:
```bash
python wiz.py
```
This will prompt you to enter a video file path or drag and drop a file.

### Specify input file:
```bash
python wiz.py -f video.mp4
```

### Specify output location:
```bash
python wiz.py -f video.mp4 -o output.mp4
```

### Use a different Whisper model:
```bash
# Faster but less accurate
python wiz.py -f video.mp4 -m tiny

# More accurate but slower
python wiz.py -f video.mp4 -m large
```

## How it works

1. The script transcribes your video using Whisper
2. The transcript opens in your default text editor (or vim)
3. Delete any words or sentences you want to remove from the video
4. Save and exit the editor
5. The script processes the video, cutting out the parts you deleted
6. The edited video is saved in your current directory

## Notes

- By default, the edited video is saved to your current working directory
- The script uses your $EDITOR environment variable, or falls back to vim
- For best results, delete more on 1 line, not less, deleting just 1 word might
  leave artifacts

## TODO

* Investigate crossfading for odd artifact removal
