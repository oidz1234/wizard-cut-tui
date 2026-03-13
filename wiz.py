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

# --- Standalone functions for reuse by preview threads ---

def compute_deleted_segment_ids(original_content: str, edited_content: str,
                                transcript_segments: list) -> Set[int]:
    """Diff original vs edited transcript content and return indices of deleted segments."""
    orig_lines = [line for line in original_content.split('\n') if not line.startswith('#')]
    edited_lines = [line for line in edited_content.split('\n') if not line.startswith('#')]
    orig_text = '\n'.join(orig_lines)
    edited_text = '\n'.join(edited_lines)

    def tokenize(text):
        silence_pattern = r'(\[SILENCE-\d+ \d+\.\d+s\])'
        protected = re.sub(silence_pattern, lambda m: m.group(0).replace(' ', '█'), text)
        tokens = re.split(r'\s+', protected)
        return [t.replace('█', ' ') for t in tokens if t]

    orig_tokens = tokenize(orig_text)
    edited_tokens = tokenize(edited_text)
    diffs = difflib.ndiff(orig_tokens, edited_tokens)

    deleted_tokens = [diff[2:] for diff in diffs if diff.startswith('- ')]

    deleted_segment_ids = set()
    for token in deleted_tokens:
        if '[SILENCE-' in token:
            match = re.search(r'\[SILENCE-(\d+)', token)
            if match:
                silence_id = int(match.group(1))
                for i, seg in enumerate(transcript_segments):
                    if seg.is_silence and seg.id == silence_id:
                        deleted_segment_ids.add(i)
                        break
            continue

        token = token.strip()
        for i, seg in enumerate(transcript_segments):
            if not seg.is_silence and seg.word.strip() == token:
                deleted_segment_ids.add(i)
                break

        if token and all(i not in deleted_segment_ids for i, seg in enumerate(transcript_segments)
                        if not seg.is_silence and seg.word.strip() == token):
            clean_token = re.sub(r'[^\w\s]', '', token.lower())
            if clean_token:
                for i, seg in enumerate(transcript_segments):
                    if seg.is_silence:
                        continue
                    if clean_token == re.sub(r'[^\w\s]', '', seg.word.lower()):
                        deleted_segment_ids.add(i)
                        break

    return deleted_segment_ids


def merge_into_cut_regions(deleted_segment_ids: Set[int],
                           transcript_segments: list) -> List[CutRegion]:
    """Merge deleted segment IDs into contiguous CutRegion objects."""
    cut_segs = sorted([transcript_segments[i] for i in deleted_segment_ids], key=lambda s: s.start)
    if not cut_segs:
        return []
    regions = []
    start, end, text = cut_segs[0].start, cut_segs[0].end, cut_segs[0].word
    for seg in cut_segs[1:]:
        if seg.start <= end + 0.3:
            end = max(end, seg.end)
            text += " " + seg.word
        else:
            regions.append(CutRegion(start, end, text))
            start, end, text = seg.start, seg.end, seg.word
    regions.append(CutRegion(start, end, text))
    return regions


def compute_keep_segments(cut_regions: list, video_duration: float) -> list:
    """Invert cut regions into a list of {'start', 'end'} segments to keep."""
    segments = []
    current = 0.0
    for region in sorted(cut_regions, key=lambda x: x.start_time):
        if region.start_time > current:
            segments.append({'start': current, 'end': region.start_time})
        current = region.end_time
    if current < video_duration:
        segments.append({'start': current, 'end': video_duration})
    return segments


def generate_edl_file(video_path: str, keep_segments: list, edl_path: str):
    """Write an mpv EDL file that plays only the kept segments."""
    abs_path = os.path.abspath(video_path)
    with open(edl_path, 'w') as f:
        f.write("# mpv EDL v0\n")
        for seg in keep_segments:
            f.write(f"{abs_path},{seg['start']},{seg['end'] - seg['start']}\n")


def original_to_edl_time(timestamp: float, keep_segments: list) -> Optional[float]:
    """Translate an original-video timestamp to EDL-timeline position."""
    edl_offset = 0.0
    for seg in keep_segments:
        if seg['start'] <= timestamp <= seg['end']:
            return edl_offset + (timestamp - seg['start'])
        edl_offset += seg['end'] - seg['start']
    return None


def edl_to_original_time(edl_time: float, keep_segments: list) -> Optional[float]:
    """Translate an EDL-timeline position back to original-video timestamp."""
    edl_offset = 0.0
    for seg in keep_segments:
        seg_duration = seg['end'] - seg['start']
        if edl_time <= edl_offset + seg_duration:
            return seg['start'] + (edl_time - edl_offset)
        edl_offset += seg_duration
    return None


# --- Preview classes ---

class MpvPreviewController:
    """Controls an mpv instance via JSON IPC over a Unix socket"""
    def __init__(self, video_path: str, socket_path: str):
        self.video_path = video_path
        self.socket_path = socket_path
        self.process = None
        self._sock = None
        self._lock = threading.Lock()

    def start(self):
        """Launch mpv paused with IPC socket"""
        self.process = subprocess.Popen([
            'mpv',
            '--input-ipc-server=' + self.socket_path,
            '--pause',
            '--keep-open=always',
            '--idle=yes',
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

    def _send_command(self, command: list, request_id: int = 1):
        """Send a command via IPC and return response data, or None on error."""
        msg = json.dumps({"command": command, "request_id": request_id}) + "\n"
        with self._lock:
            try:
                self._connect()
                self._sock.sendall(msg.encode())
                self._sock.settimeout(0.5)
                buf = b""
                while True:
                    chunk = self._sock.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line:
                            continue
                        try:
                            resp = json.loads(line)
                            if resp.get("request_id") == request_id:
                                self._sock.settimeout(None)
                                return resp
                        except json.JSONDecodeError:
                            pass
            except (ConnectionRefusedError, FileNotFoundError, BrokenPipeError,
                    OSError, socket.timeout):
                self._sock = None
            try:
                if self._sock:
                    self._sock.settimeout(None)
            except OSError:
                pass
        return None

    def seek(self, timestamp: float):
        """Seek mpv to a timestamp in seconds"""
        self._send_command(["seek", str(timestamp), "absolute"], request_id=0)

    def get_property(self, name: str):
        """Get an mpv property value. Returns None on error."""
        resp = self._send_command(["get_property", name])
        if resp and resp.get("error") == "success":
            return resp.get("data")
        return None

    def is_paused(self) -> Optional[bool]:
        return self.get_property("pause")

    def get_time_pos(self) -> Optional[float]:
        return self.get_property("time-pos")

    def load_file(self, path: str, mode: str = "replace"):
        """Load a file (or EDL) into mpv."""
        self._send_command(["loadfile", path, mode], request_id=2)

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
        self.follower = None       # PlaybackFollower reference for coordination
        self.save_watcher = None   # SaveWatcher reference for EDL time translation

    def run(self):
        while self.running:
            # Suppress seeking while mpv is playing (PlaybackFollower drives)
            if self.follower and self.follower.is_following:
                time.sleep(self.poll_interval)
                continue
            try:
                with open(self.cursor_file, 'r') as f:
                    content = f.read().strip()
                if content:
                    line, col = map(int, content.split(','))
                    ts = self._lookup(line, col)
                    if ts is not None:
                        # Translate to EDL timeline if EDL is active
                        seek_ts = ts
                        if self.save_watcher and self.save_watcher.current_keep_segments is not None:
                            edl_ts = original_to_edl_time(ts, self.save_watcher.current_keep_segments)
                            if edl_ts is None:
                                # Cursor is on a deleted segment, skip seeking
                                time.sleep(self.poll_interval)
                                continue
                            seek_ts = edl_ts
                        if self._last_timestamp is None or abs(seek_ts - self._last_timestamp) > 0.1:
                            self.mpv.seek(seek_ts)
                            self._last_timestamp = seek_ts
            except (FileNotFoundError, ValueError):
                pass
            time.sleep(self.poll_interval)

    def _lookup(self, line: int, col: int) -> Optional[float]:
        """Find the timestamp for a given line and column (1-based, matching vim)"""
        for entry in self.linecol_map:
            if entry['line'] == line and entry['col_start'] <= col <= entry['col_end']:
                return entry['start_time']
        line_entries = [e for e in self.linecol_map if e['line'] == line]
        if line_entries:
            closest = min(line_entries, key=lambda e: abs(e['col_start'] - col))
            return closest['start_time']
        if self.linecol_map:
            closest = min(self.linecol_map, key=lambda e: abs(e['line'] - line))
            return closest['start_time']
        return None

    def stop(self):
        self.running = False


class PlaybackFollower(threading.Thread):
    """When mpv is playing, writes target cursor position for vim to follow."""
    def __init__(self, mpv: MpvPreviewController, linecol_map: list,
                 target_file: str, poll_interval: float = 0.05):
        super().__init__(daemon=True)
        self.mpv = mpv
        self.linecol_map = linecol_map
        self.target_file = target_file
        self.poll_interval = poll_interval
        self.running = True
        self.is_following = False
        self.save_watcher = None  # SaveWatcher reference for EDL time translation

    def run(self):
        while self.running:
            try:
                paused = self.mpv.is_paused()
                if paused is False:  # explicitly playing
                    time_pos = self.mpv.get_time_pos()
                    if time_pos is not None:
                        # Translate from EDL time to original time if needed
                        orig_time = time_pos
                        if self.save_watcher and self.save_watcher.current_keep_segments is not None:
                            orig_time = edl_to_original_time(time_pos, self.save_watcher.current_keep_segments)
                            if orig_time is None:
                                time.sleep(self.poll_interval)
                                continue
                        lc = self._timestamp_to_linecol(orig_time)
                        if lc[0] is not None:
                            with open(self.target_file, 'w') as f:
                                f.write(f"{lc[0]},{lc[1]}")
                            self.is_following = True
                    time.sleep(self.poll_interval)
                else:
                    if self.is_following:
                        try:
                            os.unlink(self.target_file)
                        except FileNotFoundError:
                            pass
                        self.is_following = False
                    time.sleep(0.2)  # poll less when paused
            except Exception:
                time.sleep(self.poll_interval)

    def _timestamp_to_linecol(self, timestamp: float) -> Tuple[Optional[int], Optional[int]]:
        """Binary search linecol_map for entry with start_time <= timestamp."""
        if not self.linecol_map:
            return None, None
        lo, hi = 0, len(self.linecol_map) - 1
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.linecol_map[mid]['start_time'] <= timestamp:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        entry = self.linecol_map[best]
        return entry['line'], entry['col_start']

    def stop(self):
        self.running = False
        try:
            os.unlink(self.target_file)
        except FileNotFoundError:
            pass


class SaveWatcher(threading.Thread):
    """Watches for file saves and regenerates EDL for NLE preview."""
    def __init__(self, signal_file: str, editor_file: str,
                 original_content: str, transcript_segments: list,
                 video_path: str, video_duration: float,
                 edl_path: str, mpv: MpvPreviewController,
                 poll_interval: float = 0.3):
        super().__init__(daemon=True)
        self.signal_file = signal_file
        self.editor_file = editor_file
        self.original_content = original_content
        self.transcript_segments = transcript_segments
        self.video_path = video_path
        self.video_duration = video_duration
        self.edl_path = edl_path
        self.mpv = mpv
        self.poll_interval = poll_interval
        self.running = True
        self.current_keep_segments = None  # None = raw video, list = EDL active

    def run(self):
        while self.running:
            try:
                if os.path.exists(self.signal_file):
                    os.unlink(self.signal_file)
                    self._on_save()
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def _on_save(self):
        """Regenerate EDL from current editor state."""
        try:
            with open(self.editor_file, 'r') as f:
                edited_content = f.read()
        except FileNotFoundError:
            return

        deleted_ids = compute_deleted_segment_ids(
            self.original_content, edited_content, self.transcript_segments)

        if not deleted_ids:
            if self.current_keep_segments is not None:
                self.mpv.load_file(str(self.video_path))
                self.current_keep_segments = None
            return

        cut_regions = merge_into_cut_regions(deleted_ids, self.transcript_segments)
        keep_segments = compute_keep_segments(cut_regions, self.video_duration)

        if not keep_segments:
            return

        generate_edl_file(str(self.video_path), keep_segments, self.edl_path)
        self.mpv.load_file(self.edl_path)
        self.current_keep_segments = keep_segments

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
    
    def _generate_vim_preview_script(self, cursor_file: str, target_file: str, save_signal: str) -> Path:
        """Generate a Vim/Neovim script for bidirectional sync and save detection"""
        script_path = self.session_dir / "wizcut_preview.vim"
        with open(script_path, 'w') as f:
            f.write('" WizardCut live preview — auto-generated\n')
            f.write(f"let s:cursor_file = '{cursor_file}'\n")
            f.write(f"let s:target_file = '{target_file}'\n")
            f.write(f"let s:save_signal = '{save_signal}'\n")
            f.write("let s:following_playback = 0\n\n")
            # Cursor reporting (suppressed during playback follow)
            f.write("augroup WizardCutPreview\n")
            f.write("  autocmd!\n")
            f.write("  autocmd CursorMoved,CursorMovedI * if !s:following_playback | call writefile([line('.') . ',' . col('.')], s:cursor_file) | endif\n")
            f.write("  autocmd BufWritePost <buffer> call writefile(['1'], s:save_signal)\n")
            f.write("augroup END\n\n")
            # Timer: follow mpv playback position
            f.write("function! s:CheckPlaybackTarget(timer_id)\n")
            f.write("  if !filereadable(s:target_file)\n")
            f.write("    let s:following_playback = 0\n")
            f.write("    return\n")
            f.write("  endif\n")
            f.write("  let l:content = readfile(s:target_file)\n")
            f.write("  if empty(l:content) || l:content[0] == ''\n")
            f.write("    return\n")
            f.write("  endif\n")
            f.write("  let l:parts = split(l:content[0], ',')\n")
            f.write("  if len(l:parts) < 2\n")
            f.write("    return\n")
            f.write("  endif\n")
            f.write("  let s:following_playback = 1\n")
            f.write("  call cursor(str2nr(l:parts[0]), str2nr(l:parts[1]))\n")
            f.write("  call timer_start(50, {-> execute('let s:following_playback = 0')})\n")
            f.write("endfunction\n\n")
            f.write("let s:playback_timer = timer_start(50, function('s:CheckPlaybackTarget'), {'repeat': -1})\n")
            f.write("autocmd VimLeave * call timer_stop(s:playback_timer)\n")
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
        """Open editor with bidirectional mpv preview and NLE playback"""
        if not shutil.which('mpv'):
            console.print("[red]Error: mpv not found. Install mpv for live preview.[/red]")
            return False

        cursor_file = f"/tmp/wizcut_{self.session_id}_cursor"
        target_file = f"/tmp/wizcut_{self.session_id}_target"
        save_signal = f"/tmp/wizcut_{self.session_id}_save_signal"
        mpv_socket = f"/tmp/wizcut_{self.session_id}_mpv.sock"
        edl_path = str(self.session_dir / "preview.edl")

        # Start mpv
        mpv = MpvPreviewController(self.video_path, mpv_socket)
        mpv.start()
        console.print("[green]mpv preview window opened (paused)[/green]")

        # Wait for mpv socket
        for _ in range(20):
            if os.path.exists(mpv_socket):
                break
            time.sleep(0.1)

        # Get video duration for EDL generation
        video_duration = self.get_video_duration()

        # Start all threads
        watcher = CursorWatcher(cursor_file, self.linecol_map, mpv)
        follower = PlaybackFollower(mpv, self.linecol_map, target_file)
        save_watch = SaveWatcher(
            signal_file=save_signal,
            editor_file=str(self.editor_file),
            original_content=self.original_content,
            transcript_segments=self.transcript_segments,
            video_path=str(self.video_path),
            video_duration=video_duration,
            edl_path=edl_path,
            mpv=mpv,
        )

        # Wire cross-references
        watcher.follower = follower
        watcher.save_watcher = save_watch
        follower.save_watcher = save_watch

        watcher.start()
        follower.start()
        save_watch.start()

        try:
            editor_base = os.path.basename(editor)
            if 'vim' in editor_base or 'nvim' in editor_base:
                vim_script = self._generate_vim_preview_script(cursor_file, target_file, save_signal)
                cmd = [editor, '-S', str(vim_script), str(self.editor_file)]
                console.print("[green]Live preview: cursor syncs both ways. Save (:w) to update NLE preview.[/green]")
            else:
                cmd = [editor, str(self.editor_file)]
                console.print("[yellow]Warning: Full sync requires vim or nvim. mpv open but limited.[/yellow]")

            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error opening editor: {e}[/red]")
            return False
        finally:
            watcher.stop()
            follower.stop()
            save_watch.stop()
            mpv.stop()
            for f in [cursor_file, target_file, save_signal]:
                try:
                    os.unlink(f)
                except FileNotFoundError:
                    pass
    
    def detect_word_level_changes(self) -> Set[int]:
        """Detect deleted segments using word-level diffing"""
        with open(self.editor_file, 'r') as f:
            edited_content = f.read()
        return compute_deleted_segment_ids(
            self.original_content, edited_content, self.transcript_segments)
    
    def find_segments_to_cut(self) -> bool:
        """Identify segments to cut based on editor deletions"""
        deleted_segment_ids = self.detect_word_level_changes()

        if not deleted_segment_ids:
            console.print("[yellow]No content was deleted. No changes will be made to the video.[/yellow]")
            return False

        for idx in deleted_segment_ids:
            self.transcript_segments[idx].deleted = True

        console.print(f"\n[bold]Detected {len(deleted_segment_ids)} deleted segments:[/bold]")
        for idx in sorted(deleted_segment_ids):
            segment = self.transcript_segments[idx]
            console.print(f"  - {format_time(segment.start)}: '{segment.word}'")

        self.cut_regions = merge_into_cut_regions(deleted_segment_ids, self.transcript_segments)

        if self.cut_regions:
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
        video_duration = self.get_video_duration()
        segments_to_keep = compute_keep_segments(self.cut_regions, video_duration)
        
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
