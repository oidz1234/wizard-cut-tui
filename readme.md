# WizardCut TUI

Edit videos by deleting text. Whisper transcribes your video, you delete what you don't want, FFmpeg cuts it out.

## Requirements

- Python 3
- FFmpeg
- mpv (for live preview)
- vim or neovim

## Installation

```bash
pip install openai-whisper rich
```

## Usage

```bash
# Basic - transcribe, edit, cut
python wiz.py -f video.mp4

# Use a faster/smaller model
python wiz.py -f video.mp4 -m tiny

# Specify output path
python wiz.py -f video.mp4 -o output.mp4

# Disable live preview
python wiz.py -f video.mp4 --no-preview
```

## How it works

1. Whisper transcribes your video with word-level timestamps
2. The transcript opens in vim/nvim
3. Delete any words, sentences, or silence markers you want to remove
4. Save and quit - the video is processed with the cuts applied

## Live Preview

Preview is enabled by default when mpv and vim/nvim are available. An mpv window opens alongside your editor:

- **Move your cursor** through the transcript - mpv seeks to that timestamp
- **Press F5** to play/pause - the cursor follows playback in real-time
- **Save (:w)** to update the preview - deleted segments are skipped during playback
- **Press ?** for controls help

Insert mode is disabled to prevent accidental text addition.

## Notes

- For best results, delete whole words or sentences rather than single characters
- The default Whisper model is `medium` - use `tiny` for speed or `large` for accuracy
- Edited video is saved to your current directory as `{name}_edited.{ext}`
