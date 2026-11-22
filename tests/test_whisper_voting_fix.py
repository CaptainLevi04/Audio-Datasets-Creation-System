import numpy as np
import pandas as pd
import soundfile as sf
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path("Audio-Datasets-Creation-System").resolve()))
from speech_corpus.segmentation import compute_unified_voting_grid
from student1_standalone import compute_unified_voting_grid as standalone_voting_grid

def test_whisper_voting():
    print("=== RUNNING RIGOROUS WHISPER VOTING TESTS ===")
    
    sr = 16000
    t = np.linspace(0, 4, 4 * sr, endpoint=False)
    audio = np.zeros(4 * sr, dtype=np.float32)
    audio[sr:3*sr] = 0.5 * np.sin(2 * np.pi * 440 * t[sr:3*sr])
    
    # Also check if real audio file exists
    real_audio_path = Path("Audio-Datasets-Creation-System/data/demo_run/audio.wav")
    real_audio = None
    real_sr = 16000
    if real_audio_path.exists():
        real_audio, real_sr = sf.read(str(real_audio_path))
        if len(real_audio.shape) > 1:
            real_audio = real_audio.mean(axis=1)
        real_audio = real_audio[:10 * real_sr].astype(np.float32) # first 10s
    
    for name, voting_fn in [("Modular (speech_corpus.segmentation)", compute_unified_voting_grid),
                            ("Standalone (student1_standalone)", standalone_voting_grid)]:
        print(f"\n--- Testing {name} ---")
        
        # ----------------------------------------------------
        # TEST 1: WITHOUT word_timestamps (word_timestamps=None)
        # ----------------------------------------------------
        pauses_df, grid_df, timeline_df = voting_fn(audio, sr=sr, word_timestamps=None)
        
        # 1. Whisper must contribute strictly 0 votes (NOT silence votes)
        assert (grid_df["whisper_vote"] == 0).all(), "FAILURE: whisper_vote contains non-zero votes when word_timestamps=None!"
        print("  [PASS] 1. Whisper votes are strictly 0 for all bins when word_timestamps=None.")
        
        # 2. Denominator and number of detectors must be exactly 2
        assert (grid_df["num_detectors"] == 2).all(), f"FAILURE: Expected num_detectors=2, got {grid_df['num_detectors'].unique()}"
        assert grid_df.attrs["whisper_available"] is False, "FAILURE: grid_df.attrs['whisper_available'] is not False"
        assert grid_df.attrs["num_detectors"] == 2, "FAILURE: grid_df.attrs['num_detectors'] != 2"
        print("  [PASS] 2. num_detectors is 2 and metadata attributes correctly record whisper unavailable.")
        
        # 3. Maximum total votes cannot exceed 2 (only Energy and Silero can vote)
        assert grid_df["vote_count"].max() <= 2, f"FAILURE: vote_count exceeds 2: {grid_df['vote_count'].max()}"
        print("  [PASS] 3. Total vote count is strictly <= 2 across all bins.")
        
        # 4. Confidence calculation must use 2 as denominator: total_votes / 2.0
        expected_conf = np.round(grid_df["vote_count"] / 2.0, 3)
        assert np.allclose(grid_df["confidence"], expected_conf), "FAILURE: Confidence does not use num_detectors=2 as denominator!"
        assert all(str(row.voting_agreement).endswith("/2") for _, row in grid_df.iterrows()), "FAILURE: voting_agreement denominator is not /2"
        assert (grid_df["whisper_status"] == "UNAVAILABLE").all(), "FAILURE: whisper_status is not UNAVAILABLE"
        print("  [PASS] 4. Confidence and voting_agreement correctly use denominator 2 and whisper_status='UNAVAILABLE'.")
        
        # 5. Check candidate pause output table
        if not pauses_df.empty:
            assert pauses_df.attrs["whisper_available"] is False
            assert pauses_df.attrs["num_detectors"] == 2
            assert (pauses_df["whisper_status"] == "UNAVAILABLE").all()
            assert (pauses_df["num_detectors"] == 2).all()
            assert (pauses_df["confidence"] <= 1.0).all()
            assert all(str(v).endswith("/2") for v in pauses_df["voting_agreement"])
            print("  [PASS] 5. pauses_df correctly reflects num_detectors=2, whisper_status='UNAVAILABLE', and /2 agreement.")

        # ----------------------------------------------------
        # TEST 2: WITH valid word_timestamps
        # ----------------------------------------------------
        words = [{"start": 1.0, "end": 3.0, "word": "speech_segment"}]
        pauses_df_w, grid_df_w, timeline_df_w = voting_fn(audio, sr=sr, word_timestamps=words)
        
        # 1. Denominator must now be 3
        assert (grid_df_w["num_detectors"] == 3).all(), "FAILURE: num_detectors should be 3 when Whisper timestamps provided"
        assert grid_df_w.attrs["whisper_available"] is True, "FAILURE: whisper_available should be True"
        assert grid_df_w.attrs["num_detectors"] == 3, "FAILURE: num_detectors attr should be 3"
        assert (grid_df_w["whisper_status"] == "AVAILABLE").all(), "FAILURE: whisper_status should be AVAILABLE"
        print("  [PASS] 6. With word_timestamps, num_detectors is 3 and whisper_status is AVAILABLE.")
        
        # 2. Whisper votes: 0 inside word [1.0, 3.0], 1 outside (silence complement)
        inside_word = grid_df_w[(grid_df_w["start"] >= 1.1) & (grid_df_w["end"] <= 2.9)]
        outside_word = grid_df_w[(grid_df_w["end"] <= 0.9) | (grid_df_w["start"] >= 3.1)]
        assert (inside_word["whisper_vote"] == 0).all(), "FAILURE: Whisper voted silence inside word"
        assert (outside_word["whisper_vote"] == 1).all(), "FAILURE: Whisper did not vote silence outside word"
        print("  [PASS] 7. Whisper votes correctly: 0 during words, 1 during silence outside words.")
        
        # 3. Confidence calculation must use 3 as denominator
        expected_conf_w = np.round(grid_df_w["vote_count"] / 3.0, 3)
        assert np.allclose(grid_df_w["confidence"], expected_conf_w), "FAILURE: Confidence does not use num_detectors=3"
        assert all(str(row.voting_agreement).endswith("/3") for _, row in grid_df_w.iterrows()), "FAILURE: voting_agreement not /3"
        print("  [PASS] 8. Confidence and voting_agreement correctly use denominator 3 when Whisper participates.")

        # ----------------------------------------------------
        # TEST 3: Real Audio File Check
        # ----------------------------------------------------
        if real_audio is not None:
            _, g_real_no_w, _ = voting_fn(real_audio, sr=real_sr, word_timestamps=None)
            assert (g_real_no_w["whisper_vote"] == 0).all()
            assert (g_real_no_w["num_detectors"] == 2).all()
            assert g_real_no_w["vote_count"].max() <= 2
            
            _, g_real_w, _ = voting_fn(real_audio, sr=real_sr, word_timestamps=[{"start": 2.0, "end": 4.0, "word": "sample"}])
            assert (g_real_w["num_detectors"] == 3).all()
            assert (g_real_w["whisper_status"] == "AVAILABLE").all()
            print("  [PASS] 9. Real audio sample validation confirmed for both missing and present Whisper passes.")

    print("\n>>> ALL VALIDATION REQUIREMENTS VERIFIED AND PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_whisper_voting()
