#!/usr/bin/env python3
"""
WizardCut TUI 

This script transcribes your video, opens the transcript in your preferred editor,
and then processes the video by removing any parts you deleted in the editor.
"""

import os
import sys
import json
import uuid
import tempfile
import shutil
import subprocess
import time
import argparse
import difflib
import re
import socket
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

try:
    import whisper
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
except ImportError:
    print("Required packages not found. Please install them with:")
    print("pip install openai-whisper rich")
    sys.exit(1)

# Initialize rich console
console = Console()

# Configuration
TEMP_DIR = Path(tempfile.gettempdir()) / "wizardcut-tui"
os.makedirs(TEMP_DIR, exist_ok=True)

class TranscriptSegment:
    """Represents a word or silence segment in the transcript"""
    def __init__(self, word: str, start: float, end: float, is_silence: bool = False, duration: float = None, id: int = None):
        self.word = word
        self.start = start
        self.end = end
        self.is_silence = is_silence
        self.duration = duration
        self.id = id  # Unique identifier for tracking
        self.deleted = False  # Whether this segment is marked for deletion

    def __str__(self) -> str:
        if self.is_silence:
            return f"[SILENCE-{self.id} {self.duration:.1f}s]"
        else:
            return self.word

class CutRegion:
    """Represents a region to cut from the video"""
    def __init__(self, start_time: float, end_time: float, text: str):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
    
    def __str__(self) -> str:
        return f"{format_time(self.start_time)} - {format_time(self.end_time)}: {self.text}"

class MpvPreviewController:
    """Controls an mpv instance via JSON IPC over a Unix socket"""
    def __init__(self, video_path: str, socket_path: str):
        self.video_path = video_path
        self.socket_path = socket_path
        self.process = None
        self._sock = None

    def start(self):
        """Launch mpv paused with IPC socket"""
        self.process = subprocess.Popen([
            'mpv',
            '--input-ipc-server=' + self.socket_path,
            '--pause',
            '--keep-open=yes',
            '--no-terminal',
            '--force-window=yes',
            '--osd-level=1',
            str(self.video_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _connect(self):
        """Establish persistent connection to mpv socket"""
        if self._sock is None:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.connect(self.socket_path)

    def seek(self, timestamp: float):
        """Seek mpv to a timestamp in seconds"""
        cmd = json.dumps({"command": ["seek", timestamp, "absolute"]}) + "\n"
        try:
            self._connect()
            self._sock.sendall(cmd.encode())
            # Drain response to avoid buffer buildup
            self._sock.setblocking(False)
            try:
                self._sock.recv(4096)
            except BlockingIOError:
                pass
            self._sock.setblocking(True)
        except (ConnectionRefusedError, FileNotFoundError, BrokenPipeError, OSError):
            # Socket not ready or connection lost — reconnect next time
            self._sock = None

    def stop(self):
        """Terminate mpv and clean up"""
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass


class CursorWatcher(threading.Thread):
    """Watches a cursor position file and seeks mpv to the corresponding timestamp"""
    def __init__(self, cursor_file: str, linecol_map: list, mpv: MpvPreviewController, poll_interval: float = 0.1):
        super().__init__(daemon=True)
        self.cursor_file = cursor_file
        self.linecol_map = linecol_map
        self.mpv = mpv
        self.poll_interval = poll_interval
        self.running = True
        self._last_timestamp = None

    def run(self):
        while self.running:
            try:
                with open(self.cursor_file, 'r') as f:
                    content = f.read().strip()
                if content:
                    line, col = map(int, content.split(','))
                    ts = self._lookup(line, col)
                    if ts is not None and (self._last_timestamp is None or abs(ts - self._last_timestamp) > 0.1):
                        self.mpv.seek(ts)
                        self._last_timestamp = ts
            except (FileNotFoundError, ValueError):
                pass
            time.sleep(self.poll_interval)

    def _lookup(self, line: int, col: int) -> Optional[float]:
        """Find the timestamp for a given line and column (1-based, matching vim)"""
        # Exact match: find segment on this line whose col range contains the cursor
        for entry in self.linecol_map:
            if entry['line'] == line and entry['col_start'] <= col <= entry['col_end']:
                return entry['start_time']
        # Fallback: closest segment on this line
        line_entries = [e for e in self.linecol_map if e['line'] == line]
        if line_entries:
            closest = min(line_entries, key=lambda e: abs(e['col_start'] - col))
            return closest['start_time']
        # Fallback: nearest line
        if self.linecol_map:
            closest = min(self.linecol_map, key=lambda e: abs(e['line'] - line))
            return closest['start_time']
        return None

    def stop(self):
        self.running = False


class WizardCutEditor:
    """Main application class for WizardCut with system editor integration"""
    def __init__(self, output_path=None, preview=False):
        self.video_path = None
        self.audio_path = None
        self.transcript_path = None
        self.session_id = str(uuid.uuid4())
        self.session_dir = TEMP_DIR / self.session_id
        os.makedirs(self.session_dir, exist_ok=True)

        self.transcript_segments = []
        self.cut_regions = []
        self.whisper_model = None
        self.editor_file = None
        self.original_content = None
        self.output_path = output_path  # Custom output path if specified
        self.preview = preview  # Enable live mpv preview

        # For tracking word positions in the editor
        self.word_index_map = {}  # Maps word positions in editor to transcript segments
        self.segment_positions = []  # Track position of each segment in the editor
        self.linecol_map = []  # Maps (line, col) in editor to timestamps for preview
    
    def load_whisper_model(self, model_size: str = "medium") -> None:
        """Load the Whisper model"""
        with console.status(f"Loading Whisper {model_size} model..."):
            self.whisper_model = whisper.load_model(model_size)
            console.print(f"[green]Loaded Whisper {model_size} model[/green]")
    
    def load_video(self, video_path: str) -> bool:
        """Load a video file"""
        if not os.path.exists(video_path):
            console.print(f"[red]Error: File not found: {video_path}[/red]")
            return False
        
        self.video_path = video_path
        console.print(f"[green]Loaded video: {os.path.basename(video_path)}[/green]")
        return True
    
    def extract_audio(self) -> bool:
        """Extract audio from video for transcription"""
        self.audio_path = self.session_dir / "audio.wav"
        
        try:
            with console.status("Extracting audio from video..."):
                # Use ffmpeg to extract audio
                subprocess.run([
                    'ffmpeg', '-y', '-i', str(self.video_path), 
                    '-vn', '-acodec', 'pcm_s16le', 
                    '-ar', '16000', '-ac', '1', str(self.audio_path)
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error extracting audio: {e}[/red]")
            return False
    
    def get_video_duration(self) -> float:
        """Get video duration in seconds using FFprobe"""
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(self.video_path)
        ], capture_output=True, text=True, check=True)
        
        return float(result.stdout.strip())
    
    def transcribe_audio(self) -> bool:
        """Transcribe audio using Whisper"""
        try:
            if self.whisper_model is None:
                self.load_whisper_model()
            
            with console.status("[bold yellow]🪄 Transcribing audio with Whisper...[/bold yellow]"):
                # Get video duration for progress estimation
                video_duration = self.get_video_duration()
                
                # Transcribe with word timestamps
                result = self.whisper_model.transcribe(
                    str(self.audio_path),
                    word_timestamps=True,
                    language="en"  # Can be modified or auto-detected
                )
                
                # Process words with timestamps and detect silence
                self.transcript_segments = []
                prev_end_time = 0
                silence_threshold = 1.0  # Silence threshold in seconds
                segment_id = 0  # Unique ID for each segment
                
                for segment in result["segments"]:
                    for word_info in segment["words"]:
                        current_start_time = word_info['start']
                        
                        # Check for silence between words
                        silence_duration = current_start_time - prev_end_time
                        if silence_duration >= silence_threshold:
                            self.transcript_segments.append(TranscriptSegment(
                                word=f"[SILENCE-{segment_id} {silence_duration:.1f}s]",
                                start=prev_end_time,
                                end=current_start_time,
                                is_silence=True,
                                duration=silence_duration,
                                id=segment_id
                            ))
                            segment_id += 1
                        
                        # Add the actual word
                        self.transcript_segments.append(TranscriptSegment(
                            word=word_info['word'],
                            start=word_info['start'],
                            end=word_info['end'],
                            is_silence=False,
                            id=segment_id
                        ))
                        segment_id += 1
                        
                        prev_end_time = word_info['end']
                
                # Check for silence at the end of the video
                if video_duration - prev_end_time >= silence_threshold:
                    self.transcript_segments.append(TranscriptSegment(
                        word=f"[SILENCE-{segment_id} {video_duration - prev_end_time:.1f}s]",
                        start=prev_end_time,
                        end=video_duration,
                        is_silence=True,
                        duration=video_duration - prev_end_time,
                        id=segment_id
                    ))
                
                # Save transcript data for later reference
                self.transcript_path = self.session_dir / "transcript.json"
                with open(self.transcript_path, 'w') as f:
                    json.dump([{
                        'id': s.id,
                        'word': s.word,
                        'start': s.start,
                        'end': s.end,
                        'is_silence': s.is_silence,
                        'duration': s.duration if s.is_silence else None
                    } for s in self.transcript_segments], f)
                
                return True
        except Exception as e:
            console.print(f"[red]Error transcribing audio: {e}[/red]")
            return False
    
    def create_editor_file(self) -> str:
        """Create a file for editing, tracking word positions"""
        self.editor_file = self.session_dir / "transcript_edit.txt"
        self.word_index_map = {}
        self.segment_positions = []
        
        with open(self.editor_file, 'w') as f:
            # Write header with instructions
            f.write("# WizardCut Editor - Video Transcript\n")
            f.write("#\n")
            f.write("# INSTRUCTIONS:\n")
            f.write("# - DELETE any words, sentences, or silences you want to remove from the video\n")
            f.write("# - DO NOT Delete half a word or something silly, it's like dividing by zero\n")
            f.write("# - DO NOT add new text\n")
            f.write("# - When you're finished, save and close the editor\n")
            f.write("# - The script will process the video based on what you removed\n")
            f.write("#\n\n")
            
            # Write transcript text for editing
            # Track character positions in the file for each segment
            char_position = f.tell()
            
            current_paragraph = ""
            last_segment_was_silence = False
            
            for i, segment in enumerate(self.transcript_segments):
                self.segment_positions.append(char_position)
                
                if segment.is_silence:
                    # Write current paragraph before silence marker
                    if current_paragraph:
                        f.write(current_paragraph.strip() + "\n\n")
                        char_position = f.tell()
                        current_paragraph = ""
                    
                    # Write silence marker
                    silence_text = str(segment)
                    f.write(silence_text + "\n\n")
                    char_position = f.tell()
                    last_segment_was_silence = True
                else:
                    # Add space before word unless it's punctuation or first word after silence
                    if not segment.word.strip().startswith((',', '.', '!', '?', ':', ';')) and not last_segment_was_silence and current_paragraph:
                        current_paragraph += " "
                    
                    # Add word to current paragraph
                    self.word_index_map[len(current_paragraph)] = i
                    current_paragraph += segment.word
                    last_segment_was_silence = False
                    
                    # If we have a sentence end, consider ending paragraph
                    if segment.word.strip().endswith(('.', '!', '?')) and len(current_paragraph) > 60:
                        f.write(current_paragraph.strip() + "\n\n")
                        char_position = f.tell()
                        current_paragraph = ""
            
            # Write any remaining paragraph text
            if current_paragraph:
                f.write(current_paragraph.strip() + "\n")
        
        # Save original content for comparison
        with open(self.editor_file, 'r') as f:
            self.original_content = f.read()
        
        # Create a map file for debugging or for precise transcript segment referencing
        mapfile = self.session_dir / "transcript_map.txt"
        with open(mapfile, 'w') as f:
            for pos, idx in sorted(self.word_index_map.items()):
                segment = self.transcript_segments[idx]
                f.write(f"Position {pos}: Segment {idx}, Word: '{segment.word}', Time: {segment.start:.2f}-{segment.end:.2f}\n")

        # Build line-column map for preview sync
        if self.preview:
            self._build_linecol_map()

        return str(self.editor_file)

    def _build_linecol_map(self):
        """Build a mapping from (line, col) in the editor file to segment timestamps"""
        self.linecol_map = []
        with open(self.editor_file, 'r') as f:
            lines = f.readlines()

        # Match words sequentially through the file
        word_segments = [s for s in self.transcript_segments if not s.is_silence]
        word_idx = 0

        for line_num_0, line in enumerate(lines):
            line_num = line_num_0 + 1  # vim uses 1-based line numbers
            line = line.rstrip('\n')

            if not line or line.startswith('#'):
                continue

            # Check for silence marker
            silence_match = re.match(r'\[SILENCE-(\d+) [\d.]+s\]', line)
            if silence_match:
                sid = int(silence_match.group(1))
                for seg in self.transcript_segments:
                    if seg.is_silence and seg.id == sid:
                        self.linecol_map.append({
                            'line': line_num,
                            'col_start': 1,
                            'col_end': len(line),
                            'start_time': seg.start,
                            'end_time': seg.end,
                        })
                        break
                continue

            # Regular text line — match words forward
            col = 0
            while word_idx < len(word_segments) and col < len(line):
                seg = word_segments[word_idx]
                word = seg.word.strip()
                if not word:
                    word_idx += 1
                    continue

                pos = line.find(word, col)
                if pos == -1:
                    break  # remaining words must be on later lines

                self.linecol_map.append({
                    'line': line_num,
                    'col_start': pos + 1,       # 1-based for vim
                    'col_end': pos + len(word),  # 1-based inclusive
                    'start_time': seg.start,
                    'end_time': seg.end,
                })
                col = pos + len(word)
                word_idx += 1

        # Save for debugging
        linecol_path = self.session_dir / "transcript_linecol_map.json"
        with open(linecol_path, 'w') as f:
            json.dump(self.linecol_map, f, indent=2)
    
    def _generate_vim_preview_script(self, cursor_file: str) -> Path:
        """Generate a Vim/Neovim script that reports cursor position on every move"""
        script_path = self.session_dir / "wizcut_preview.vim"
        with open(script_path, 'w') as f:
            f.write('" WizardCut live preview — auto-generated\n')
            f.write(f"let s:cursor_file = '{cursor_file}'\n")
            f.write("augroup WizardCutPreview\n")
            f.write("  autocmd!\n")
            f.write("  autocmd CursorMoved,CursorMovedI * call writefile([line('.') . ',' . col('.')], s:cursor_file)\n")
            f.write("augroup END\n")
        return script_path

    def open_in_editor(self) -> bool:
        """Open the transcript file in the system editor"""
        try:
            editor = os.environ.get('EDITOR', 'vim')
            console.print(f"[bold cyan]Opening transcript in {editor}. Delete unwanted content, then save and quit.[/bold cyan]")
            console.print(f"[yellow]Tip: Deleting any word, sentence, or silence marker will remove it from the video.[/yellow]")

            if self.preview:
                return self._open_with_preview(editor)

            subprocess.run([editor, str(self.editor_file)], check=True)
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error opening editor: {e}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            return False

    def _open_with_preview(self, editor: str) -> bool:
        """Open editor with live mpv preview synced to cursor position"""
        # Check mpv is available
        if not shutil.which('mpv'):
            console.print("[red]Error: mpv not found. Install mpv for live preview.[/red]")
            return False

        cursor_file = f"/tmp/wizcut_{self.session_id}_cursor"
        mpv_socket = f"/tmp/wizcut_{self.session_id}_mpv.sock"

        # Start mpv
        mpv = MpvPreviewController(self.video_path, mpv_socket)
        mpv.start()
        console.print("[green]mpv preview window opened (paused)[/green]")

        # Wait for mpv socket to be ready
        for _ in range(20):
            if os.path.exists(mpv_socket):
                break
            time.sleep(0.1)

        # Start cursor watcher
        watcher = CursorWatcher(cursor_file, self.linecol_map, mpv)
        watcher.start()

        try:
            # Detect vim/nvim for cursor reporting
            editor_base = os.path.basename(editor)
            if 'vim' in editor_base or 'nvim' in editor_base:
                vim_script = self._generate_vim_preview_script(cursor_file)
                cmd = [editor, '-S', str(vim_script), str(self.editor_file)]
                console.print("[green]Live preview: cursor position syncs to video timestamp[/green]")
            else:
                cmd = [editor, str(self.editor_file)]
                console.print(f"[yellow]Warning: Live cursor sync requires vim or nvim. mpv is open but won't auto-seek.[/yellow]")

            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error opening editor: {e}[/red]")
            return False
        finally:
            watcher.stop()
            mpv.stop()
            # Clean up cursor file
            try:
                os.unlink(cursor_file)
            except FileNotFoundError:
                pass
    
    def detect_word_level_changes(self) -> Set[int]:
        """Detect deleted segments using word-level diffing"""
        # Read the edited content
        with open(self.editor_file, 'r') as f:
            edited_content = f.read()
        
        # Remove comment lines from both versions for comparison
        orig_lines = [line for line in self.original_content.split('\n') if not line.startswith('#')]
        edited_lines = [line for line in edited_content.split('\n') if not line.startswith('#')]
        
        # Combine into full text for word-level comparison
        orig_text = '\n'.join(orig_lines)
        edited_text = '\n'.join(edited_lines)
        
        # Split into words for diffing
        # We need to preserve punctuation and special markers like [SILENCE]
        def tokenize(text):
            # Split by spaces but keep silence markers and punctuation intact
            silence_pattern = r'(\[SILENCE-\d+ \d+\.\d+s\])'
            # First, protect silence markers by replacing spaces within them
            protected = re.sub(silence_pattern, lambda m: m.group(0).replace(' ', '█'), text)
            # Split by whitespace
            tokens = re.split(r'\s+', protected)
            # Restore original silence markers
            tokens = [t.replace('█', ' ') for t in tokens if t]
            return tokens
        
        orig_tokens = tokenize(orig_text)
        edited_tokens = tokenize(edited_text)
        
        # Use difflib to get the differences
        diffs = difflib.ndiff(orig_tokens, edited_tokens)
        
        # Collect deletions
        deleted_tokens = []
        for diff in diffs:
            if diff.startswith('- '):
                deleted_tokens.append(diff[2:])
        
        # Find transcript segments corresponding to deleted tokens
        deleted_segment_ids = set()
        for token in deleted_tokens:
            # Handle silence markers
            if '[SILENCE-' in token:
                try:
                    # Extract silence ID from the marker
                    match = re.search(r'\[SILENCE-(\d+)', token)
                    if match:
                        silence_id = int(match.group(1))
                        # Find corresponding silence segment
                        for i, segment in enumerate(self.transcript_segments):
                            if segment.is_silence and segment.id == silence_id:
                                deleted_segment_ids.add(i)
                                break
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to parse silence marker '{token}': {e}[/yellow]")
                continue
            
            # For regular words, find exact matches or close matches
            token = token.strip()
            for i, segment in enumerate(self.transcript_segments):
                if segment.is_silence:
                    continue
                
                if segment.word.strip() == token:
                    deleted_segment_ids.add(i)
                    break
            
            # If no exact match, try fuzzy matching for words with punctuation variations
            if token and all(i not in deleted_segment_ids for i, segment in enumerate(self.transcript_segments) 
                           if not segment.is_silence and segment.word.strip() == token):
                clean_token = re.sub(r'[^\w\s]', '', token.lower())
                if clean_token:  # Only proceed if we have something to match after cleaning
                    for i, segment in enumerate(self.transcript_segments):
                        if segment.is_silence:
                            continue
                        clean_segment = re.sub(r'[^\w\s]', '', segment.word.lower())
                        if clean_token == clean_segment:
                            deleted_segment_ids.add(i)
                            break
        
        return deleted_segment_ids
    
    def find_segments_to_cut(self) -> bool:
        """Identify segments to cut based on editor deletions"""
        # Use word-level diffing to detect deleted segments
        deleted_segment_ids = self.detect_word_level_changes()
        
        if not deleted_segment_ids:
            console.print("[yellow]No content was deleted. No changes will be made to the video.[/yellow]")
            return False
        
        # Mark deleted segments
        for idx in deleted_segment_ids:
            self.transcript_segments[idx].deleted = True
        
        # Display all found deleted content for debugging
        console.print(f"\n[bold]Detected {len(deleted_segment_ids)} deleted segments:[/bold]")
        for idx in sorted(deleted_segment_ids):
            segment = self.transcript_segments[idx]
            console.print(f"  - {format_time(segment.start)}: '{segment.word}'")
        
        # Process deleted segments into cut regions
        cut_segments = [self.transcript_segments[idx] for idx in sorted(deleted_segment_ids)]
        
        # Merge adjacent segments into cut regions
        if cut_segments:
            cut_segments.sort(key=lambda s: s.start)
            
            current_region_start = cut_segments[0].start
            current_region_end = cut_segments[0].end
            current_region_text = cut_segments[0].word
            
            for i in range(1, len(cut_segments)):
                # If this segment is adjacent or close to previous (within 0.3 seconds)
                if cut_segments[i].start <= current_region_end + 0.3:
                    # Extend the current region
                    current_region_end = max(current_region_end, cut_segments[i].end)
                    current_region_text += " " + cut_segments[i].word
                else:
                    # Create a new cut region
                    self.cut_regions.append(CutRegion(
                        current_region_start,
                        current_region_end,
                        current_region_text
                    ))
                    # Start a new region
                    current_region_start = cut_segments[i].start
                    current_region_end = cut_segments[i].end
                    current_region_text = cut_segments[i].word
            
            # Add the last region
            self.cut_regions.append(CutRegion(
                current_region_start,
                current_region_end,
                current_region_text
            ))
            
            # Display the identified cut regions
            console.print(f"\n[green]Found {len(self.cut_regions)} regions to cut:[/green]")
            for i, region in enumerate(self.cut_regions, 1):
                console.print(f"  {i}. {region}")
            
            return True
        else:
            console.print("[yellow]Could not identify specific regions to cut. No changes will be made.[/yellow]")
            return False
    
    def process_video(self) -> None:
        """Process the video to remove cut regions"""
        if not self.cut_regions:
            console.print("[yellow]No regions to cut. No changes will be made to the video.[/yellow]")
            return
        
        console.print(Panel("[bold blue]Processing Video[/bold blue]"))
        
        # Determine output path
        if self.output_path:
            # Custom output path was specified
            output_path = self.output_path
            
            # If it's a directory, append the filename
            if os.path.isdir(self.output_path):
                input_filename = os.path.basename(self.video_path)
                filename_without_ext, ext = os.path.splitext(input_filename)
                output_filename = f"{filename_without_ext}_edited{ext}"
                output_path = os.path.join(self.output_path, output_filename)
        else:
            # Default: current working directory with "_edited" suffix
            input_filename = os.path.basename(self.video_path)
            filename_without_ext, ext = os.path.splitext(input_filename)
            output_filename = f"{filename_without_ext}_edited{ext}"
            output_path = os.path.join(os.getcwd(), output_filename)
        
        # Create a list of segments to keep (inverse of what to cut)
        segments_to_keep = []
        current_start = 0
        
        # Sort cut regions by start time to process in order
        sorted_regions = sorted(self.cut_regions, key=lambda x: x.start_time)
        
        for region in sorted_regions:
            # Keep segment from current_start to region start
            if region.start_time > current_start:
                segments_to_keep.append({
                    'start': current_start,
                    'end': region.start_time
                })
            
            # Update current_start to after this region
            current_start = region.end_time
        
        # Add final segment if needed
        video_duration = self.get_video_duration()
        if current_start < video_duration:
            segments_to_keep.append({
                'start': current_start,
                'end': video_duration
            })
        
        # Create temporary file for filter complex script
        filter_file = self.session_dir / "filter_complex.txt"
        with open(filter_file, 'w') as f:
            for i, segment in enumerate(segments_to_keep):
                f.write(f"[0:v]trim={segment['start']}:{segment['end']},setpts=PTS-STARTPTS[v{i}];\n")
                f.write(f"[0:a]atrim={segment['start']}:{segment['end']},asetpts=PTS-STARTPTS[a{i}];\n")
            
            # Concatenate video and audio streams
            v_stream = ''.join(f'[v{i}]' for i in range(len(segments_to_keep)))
            a_stream = ''.join(f'[a{i}]' for i in range(len(segments_to_keep)))
            
            if segments_to_keep:
                f.write(f"{v_stream}concat=n={len(segments_to_keep)}:v=1:a=0[outv];\n")
                f.write(f"{a_stream}concat=n={len(segments_to_keep)}:v=0:a=1[outa]")
            else:
                console.print("[red]Error: No segments to keep - entire video would be cut[/red]")
                return
        
        # Run FFmpeg for full quality edit
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]✨ Processing video..."),
            BarColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("Processing", total=100)
            
            try:
                # Build the FFmpeg command
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', str(self.video_path), 
                    '-filter_complex_script', str(filter_file),
                    '-map', '[outv]', '-map', '[outa]',
                    '-c:v', 'libx264', '-preset', 'medium',
                    '-c:a', 'aac', output_path
                ]
                
                # Run FFmpeg
                process = subprocess.Popen(
                    ffmpeg_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    universal_newlines=True
                )
                
                # Update progress
                while process.poll() is None:
                    progress.update(task, advance=0.5)
                    time.sleep(0.1)
                
                # Process completed
                progress.update(task, completed=100)
                
                # Check result
                if process.returncode == 0:
                    console.print(f"\n[green]✅ Video processing complete![/green]")
                    console.print(f"\nOutput saved to: [bold]{output_path}[/bold]")
                    
                    # Calculate how much was cut
                    original_duration = self.get_video_duration()
                    saved_time = sum(r.end_time - r.start_time for r in self.cut_regions)
                    console.print(f"\nOriginal duration: {format_time(original_duration)}")
                    console.print(f"Time removed: {format_time(saved_time)} ({saved_time/original_duration*100:.1f}%)")
                    console.print(f"New duration: {format_time(original_duration - saved_time)}")
                else:
                    stderr = process.stderr.read()
                    console.print(f"\n[red]Error processing video: {stderr}[/red]")
            
            except Exception as e:
                console.print(f"\n[red]Error processing video: {e}[/red]")
                return
    
    def cleanup(self) -> None:
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.session_dir)
            console.print(f"[dim]Cleaned up temporary files[/dim]")
        except Exception as e:
            console.print(f"[dim]Warning: Could not clean up all temporary files: {e}[/dim]")
    
    def run(self) -> None:
        """Main application flow"""
        # Create and edit the file
        self.create_editor_file()
        if not self.open_in_editor():
            return
        
        # Find segments to cut
        if self.find_segments_to_cut():
            # Process the video
            self.process_video()
        else:
            console.print("[yellow]No changes detected. The video will not be modified.[/yellow]")

def format_time(seconds: float) -> str:
    """Format seconds as MM:SS"""
    mins = int(seconds / 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

def get_video_path():
    """Get video file path with support for drag and drop"""
    console.print(Panel("[bold blue]WizardCut-tui[/bold blue]", subtitle="Text 'Based' Video Editor"))
    console.print("\n[bold yellow]🪄 Welcome to WizardCut![/bold yellow]")
    console.print("This application allows you to edit videos by deleting text in your editor.\n")
    
    console.print("[bold cyan]Enter path to video file or drag and drop a video file into the terminal:[/bold cyan]")
    path = input().strip()
    
    # Clean up the path (terminals often add quotes or escape characters when drag-dropping)
    path = path.strip("'\"")  # Remove quotes
    path = os.path.expanduser(path)  # Expand ~ to home directory
    
    return path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="WizardCut-tui - Text 'based' video editor")
    parser.add_argument("-f", "--file", help="Path to video file")
    parser.add_argument("-o", "--output", help="Output path (file or directory)")
    parser.add_argument("-m", "--model", choices=["tiny", "base", "small", "medium", "large"], default="medium",
                       help="Whisper model size (default: medium)")
    parser.add_argument("-p", "--preview", action="store_true",
                       help="Enable live video preview with mpv (requires mpv and vim/nvim)")
    args = parser.parse_args()
    
    try:
        # Get video file path
        video_path = args.file
        if not video_path:
            video_path = get_video_path()
        
        # Prepare output path if specified
        output_path = args.output
        if output_path:
            output_path = os.path.expanduser(output_path)
        
        app = WizardCutEditor(output_path=output_path, preview=args.preview)
        
        # Load and process the video
        if app.load_video(video_path):
            # Load the Whisper model
            app.load_whisper_model(args.model)
            
            # Extract audio
            if app.extract_audio():
                # Transcribe
                if app.transcribe_audio():
                    # Run the main process
                    app.run()
        
        # Cleanup
        app.cleanup()
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Program interrupted. Exiting...[/yellow]")
    except Exception as e:
        console.print(f"\n[red]An error occurred: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
    
    console.print("\n[bold blue]Thanks for using WizardCut![/bold blue]")

if __name__ == "__main__":
    main()
