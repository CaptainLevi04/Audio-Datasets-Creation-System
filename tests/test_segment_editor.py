import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from speech_corpus.quality_control import validate_segments_integrity
from speech_corpus.segment_editor import (
    launch_segment_editor,
    plot_segments_window,
    segment_add,
    segment_delete,
    segment_merge,
    segment_move,
    segment_split,
    sync_segments_to_disk,
)


def _sample_regions() -> pd.DataFrame:
    return pd.DataFrame([
        {"region_id": "R0001", "start": 0.0, "end": 18.0, "duration": 18.0, "label": "speech", "confidence": 0.92},
        {"region_id": "R0002", "start": 18.0, "end": 30.0, "duration": 12.0, "label": "mixed", "confidence": 0.84},
    ])


def _sample_segments() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "video_id": "vid1", "segment_id": "0001", "wav_file": "segments/segment_0001.wav",
            "start": 0.0, "end": 10.0, "duration": 10.0,
            "label": "speech", "primary_label": "speech", "label_composition": "speech:10.00",
            "source_region_ids": "R0001", "confidence": 0.92,
            "cut_type": "voting_pause_midpoint", "cut_point_time": 10.0,
            "lead_ms": 150, "trail_ms": 150, "margin_ok": 1,
            "cut_certainty": 0.95, "cut_tier": "HIGH", "voting_agreement": "2/2",
            "decision_class": "STRONG", "needs_review": 0, "review_priority": "LOW",
            "review_reason": "none", "fallback_used": 0, "plan_tag": "ok",
            "transition_status": "none", "word_check_status": "SAFE", "status": "PASSED",
            "speech_duration": 10.0, "mixed_duration": 0.0, "music_duration": 0.0, "silence_duration": 0.0,
            "expected_midpoint": 10.0, "midpoint_error_ms": 0.0, "is_verified_midpoint": 1,
            "pause_start": 9.8, "pause_end": 10.2, "transition_request_ids": "",
        },
        {
            "video_id": "vid1", "segment_id": "0002", "wav_file": "segments/segment_0002.wav",
            "start": 10.0, "end": 20.0, "duration": 10.0,
            "label": "speech", "primary_label": "speech", "label_composition": "speech:8.00|mixed:2.00",
            "source_region_ids": "R0001,R0002", "confidence": 0.90,
            "cut_type": "voting_pause_midpoint", "cut_point_time": 20.0,
            "lead_ms": 150, "trail_ms": 150, "margin_ok": 1,
            "cut_certainty": 0.92, "cut_tier": "HIGH", "voting_agreement": "2/2",
            "decision_class": "STRONG", "needs_review": 0, "review_priority": "LOW",
            "review_reason": "none", "fallback_used": 0, "plan_tag": "ok",
            "transition_status": "none", "word_check_status": "SAFE", "status": "PASSED",
            "speech_duration": 8.0, "mixed_duration": 2.0, "music_duration": 0.0, "silence_duration": 0.0,
            "expected_midpoint": 20.0, "midpoint_error_ms": 0.0, "is_verified_midpoint": 1,
            "pause_start": 19.8, "pause_end": 20.2, "transition_request_ids": "",
        },
        {
            "video_id": "vid1", "segment_id": "0003", "wav_file": "segments/segment_0003.wav",
            "start": 20.0, "end": 30.0, "duration": 10.0,
            "label": "mixed", "primary_label": "mixed", "label_composition": "mixed:10.00",
            "source_region_ids": "R0002", "confidence": 0.84,
            "cut_type": "file_edge", "cut_point_time": 30.0,
            "lead_ms": 150, "trail_ms": 0, "margin_ok": 1,
            "cut_certainty": 0.95, "cut_tier": "HIGH", "voting_agreement": "2/2",
            "decision_class": "FILE_EDGE", "needs_review": 0, "review_priority": "LOW",
            "review_reason": "none", "fallback_used": 0, "plan_tag": "ok",
            "transition_status": "none", "word_check_status": "SAFE", "status": "PASSED",
            "speech_duration": 0.0, "mixed_duration": 10.0, "music_duration": 0.0, "silence_duration": 0.0,
            "expected_midpoint": 30.0, "midpoint_error_ms": 0.0, "is_verified_midpoint": 1,
            "pause_start": np.nan, "pause_end": np.nan, "transition_request_ids": "",
        },
    ])


class TestSegmentEditor(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.audio = (np.random.default_rng(7).normal(0, 0.05, size=self.sr * 30)).astype(np.float32)
        self.regions = _sample_regions()
        self.df = _sample_segments()
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="seg_editor_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_segment_move_sync_and_unsynced(self):
        moved = segment_move(
            self.df, 1, new_start=8.5, new_end=21.5,
            sync_neighbors=True, regions_df=self.regions, total_dur=30.0,
        )
        self.assertEqual(len(moved), 3)
        self.assertAlmostEqual(moved.loc[0, "end"], 8.5, places=3)
        self.assertAlmostEqual(moved.loc[1, "start"], 8.5, places=3)
        self.assertAlmostEqual(moved.loc[1, "end"], 21.5, places=3)
        self.assertAlmostEqual(moved.loc[2, "start"], 21.5, places=3)
        self.assertEqual(moved.loc[1, "status"], "human_edited")
        self.assertEqual(moved.loc[1, "is_verified_midpoint"], 0)

        # Unsynced shrink leaves valid non-overlapping gaps, while overlap raises ValueError
        shrunk = segment_move(self.df, 1, new_start=11.0, new_end=19.0, sync_neighbors=False, total_dur=30.0)
        self.assertAlmostEqual(shrunk.loc[1, "duration"], 8.0, places=3)
        with self.assertRaises(ValueError):
            segment_move(self.df, 1, new_start=9.0, new_end=19.0, sync_neighbors=False, total_dur=30.0)

    def test_segment_split(self):
        split_df = segment_split(self.df, 1, split_time=15.0, regions_df=self.regions)
        self.assertEqual(len(split_df), 4)
        self.assertEqual(list(split_df["segment_id"]), ["0001", "0002", "0003", "0004"])
        self.assertAlmostEqual(split_df.loc[1, "start"], 10.0, places=3)
        self.assertAlmostEqual(split_df.loc[1, "end"], 15.0, places=3)
        self.assertAlmostEqual(split_df.loc[2, "start"], 15.0, places=3)
        self.assertAlmostEqual(split_df.loc[2, "end"], 20.0, places=3)
        self.assertEqual(split_df.loc[1, "cut_type"], "human_split")
        with self.assertRaises(ValueError):
            segment_split(self.df, 1, split_time=10.05)

    def test_segment_merge_next_and_prev(self):
        merged_next = segment_merge(self.df, 0, direction="next", regions_df=self.regions)
        self.assertEqual(len(merged_next), 2)
        self.assertAlmostEqual(merged_next.loc[0, "start"], 0.0, places=3)
        self.assertAlmostEqual(merged_next.loc[0, "end"], 20.0, places=3)
        self.assertAlmostEqual(merged_next.loc[0, "duration"], 20.0, places=3)
        self.assertEqual(merged_next.loc[0, "fallback_used"], 1)  # > 15s marked fallback cleanly

        merged_prev = segment_merge(self.df, 2, direction="prev", regions_df=self.regions)
        self.assertEqual(len(merged_prev), 2)
        self.assertAlmostEqual(merged_prev.loc[1, "start"], 10.0, places=3)
        self.assertAlmostEqual(merged_prev.loc[1, "end"], 30.0, places=3)

        with self.assertRaises(ValueError):
            segment_merge(self.df, 0, direction="prev")
        with self.assertRaises(ValueError):
            segment_merge(self.df, 2, direction="next")

    def test_segment_delete_remove_and_absorb(self):
        deleted = segment_delete(self.df, 1, absorb=False)
        self.assertEqual(len(deleted), 2)
        self.assertEqual(list(deleted["segment_id"]), ["0001", "0002"])
        self.assertAlmostEqual(deleted.loc[0, "end"], 10.0, places=3)
        self.assertAlmostEqual(deleted.loc[1, "start"], 20.0, places=3)

        absorbed = segment_delete(self.df, 1, absorb=True, regions_df=self.regions)
        self.assertEqual(len(absorbed), 2)
        self.assertAlmostEqual(absorbed.loc[0, "start"], 0.0, places=3)
        self.assertAlmostEqual(absorbed.loc[0, "end"], 20.0, places=3)

    def test_segment_add_and_overlap_guard(self):
        # First delete middle segment so [10, 20] is free, then add a [11, 18] music segment
        deleted = segment_delete(self.df, 1, absorb=False)
        added = segment_add(deleted, start=11.0, end=18.0, label="music", total_dur=30.0)
        self.assertEqual(len(added), 3)
        self.assertEqual(added.loc[1, "primary_label"], "music")
        self.assertAlmostEqual(added.loc[1, "duration"], 7.0, places=3)
        self.assertEqual(added.loc[1, "cut_type"], "human_added")

        # Overlapping insertion fails unless allow_overlap=True
        with self.assertRaises(ValueError):
            segment_add(self.df, start=5.0, end=15.0, label="speech")

    def test_disk_sync_and_ui_widget_flow(self):
        synced = sync_segments_to_disk(self.df, self.audio, self.tmp_dir, sr=self.sr)
        self.assertTrue((self.tmp_dir / "segments.csv").exists())
        for r in synced.itertuples():
            wp = self.tmp_dir / r.wav_file
            self.assertTrue(wp.exists())
            info = sf.info(str(wp))
            self.assertAlmostEqual(info.duration, r.duration, delta=0.02)

        # Plot smoke test
        fig = plot_segments_window(self.audio, synced, 0.0, 20.0, sr=self.sr, selected_pos=0)
        self.assertIsNotNone(fig)

        # Verify QC Gates pass on split segments synced to disk
        split_df = segment_split(synced, 1, split_time=15.0, regions_df=self.regions)
        synced_split = sync_segments_to_disk(split_df, self.audio, self.tmp_dir, sr=self.sr, old_df=synced)
        qc = validate_segments_integrity(synced_split, video_dir=self.tmp_dir, total_dur=30.0, sr=self.sr)
        self.assertTrue(qc["passed"], f"QC failed after split/sync: {qc['errors']}")

        # Restore synced 3-segment disk state before exercising interactive UI callbacks
        synced = sync_segments_to_disk(synced, self.audio, self.tmp_dir, sr=self.sr, old_df=synced_split)
        state = launch_segment_editor(
            self.audio, synced, self.tmp_dir, sr=self.sr, regions_df=self.regions, user="tester"
        )
        self.assertIn("df", state)
        self.assertEqual(len(state["df"]), 3)

        ui = state["ui"]
        # Trigger Move button on segment 0 ([0, 10] -> [0, 9.5])
        move_row = ui.children[6]
        w_start, w_end, _, b_move = move_row.children
        w_start.value = 0.0
        w_end.value = 9.5
        b_move.click()
        self.assertAlmostEqual(float(state["df"].loc[0, "end"]), 9.5, places=3)
        self.assertAlmostEqual(float(state["df"].loc[1, "start"]), 9.5, places=3)

        # Trigger Split button on segment 0 at 4.75s -> 4 segments total
        split_merge_row = ui.children[7]
        w_split, b_split, b_merge_prev, b_merge_next = split_merge_row.children
        w_split.value = 4.75
        b_split.click()
        self.assertEqual(len(state["df"]), 4)

        # Trigger Merge w/ Next -> back to 3 segments
        b_merge_next.click()
        self.assertEqual(len(state["df"]), 3)

        # Trigger Delete without absorb -> 2 segments ([9.5..20], [20..30])
        del_row = ui.children[8]
        _, b_delete = del_row.children
        b_delete.click()
        self.assertEqual(len(state["df"]), 2)

        # Trigger Add Segment into deleted [1.0, 8.0] span -> back to 3 segments
        add_row = ui.children[9]
        w_add_s, w_add_e, w_add_lbl, b_add = add_row.children
        w_add_s.value = 1.0
        w_add_e.value = 8.0
        w_add_lbl.value = "music"
        b_add.click()
        self.assertEqual(len(state["df"]), 3)
        self.assertEqual(state["df"].loc[0, "primary_label"], "music")

        # Confirm segments.csv, audio WAVs, and changes.csv audit trail match disk state
        disk_df = pd.read_csv(self.tmp_dir / "segments.csv")
        self.assertEqual(len(disk_df), 3)
        changes_df = pd.read_csv(self.tmp_dir / "changes.csv")
        self.assertEqual(list(changes_df["field"]), ["move", "split", "merge_next", "delete", "add"])


if __name__ == "__main__":
    unittest.main()
