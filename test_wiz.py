import unittest

from wiz import (
    CutRegion,
    TranscriptSegment,
    compute_deleted_segment_ids,
    compute_keep_segments,
    format_time,
    normalize_path,
)


class TranscriptDiffTests(unittest.TestCase):
    def test_deletes_contextual_duplicate_word(self):
        segments = [
            TranscriptSegment("alpha", 0.0, 0.4, id=0),
            TranscriptSegment("beta", 0.4, 0.8, id=1),
            TranscriptSegment("alpha", 0.8, 1.2, id=2),
            TranscriptSegment("[SILENCE-3 1.2s]", 1.2, 2.4, is_silence=True, duration=1.2, id=3),
            TranscriptSegment("tail", 2.4, 2.8, id=4),
        ]
        original = "alpha beta alpha\n\n[SILENCE-3 1.2s]\n\ntail\n"
        edited = "alpha beta\n\n[SILENCE-3 1.2s]\n\ntail\n"

        self.assertEqual(compute_deleted_segment_ids(original, edited, segments), {2})

    def test_punctuation_edit_does_not_cut_word(self):
        segments = [
            TranscriptSegment("Hello,", 0.0, 0.5, id=0),
            TranscriptSegment("world!", 0.5, 1.0, id=1),
        ]
        original = "Hello, world!\n"
        edited = "Hello world!\n"

        self.assertEqual(compute_deleted_segment_ids(original, edited, segments), set())

    def test_deletes_silence_marker_by_id(self):
        segments = [
            TranscriptSegment("before", 0.0, 0.5, id=0),
            TranscriptSegment("[SILENCE-1 1.5s]", 0.5, 2.0, is_silence=True, duration=1.5, id=1),
            TranscriptSegment("after", 2.0, 2.5, id=2),
        ]
        original = "before\n\n[SILENCE-1 1.5s]\n\nafter\n"
        edited = "before\n\nafter\n"

        self.assertEqual(compute_deleted_segment_ids(original, edited, segments), {1})


class SegmentMathTests(unittest.TestCase):
    def test_keep_segments_are_clamped_and_merged(self):
        regions = [
            CutRegion(8.0, 12.0, "end"),
            CutRegion(-1.0, 2.0, "start"),
            CutRegion(3.0, 5.0, "middle"),
            CutRegion(4.0, 6.0, "overlap"),
        ]

        self.assertEqual(
            compute_keep_segments(regions, 10.0),
            [
                {"start": 2.0, "end": 3.0},
                {"start": 6.0, "end": 8.0},
            ],
        )

    def test_normalize_path_handles_shell_escaped_spaces(self):
        self.assertTrue(normalize_path(r"~/some\ path/video.mp4").endswith("/some path/video.mp4"))

    def test_format_time_preserves_useful_fraction(self):
        self.assertEqual(format_time(0.5), "0:00.5")
        self.assertEqual(format_time(62.0), "1:02")


if __name__ == "__main__":
    unittest.main()
