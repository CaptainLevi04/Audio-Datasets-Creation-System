import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path("Audio-Datasets-Creation-System").resolve()))
from speech_corpus.inspector import launch_segment_inspector, sample_manual_check_window
from speech_corpus.cut_engine_v2 import cut_speech_and_mixed_segments

def test_inspector_fixes():
    print("=== TESTING INSPECTOR 4 AND MANUAL CHECK FIXES ===")
    
    sr = 16000
    audio = np.random.randn(sr * 10).astype(np.float32) * 0.05
    video_dir = Path("data/test_inspector_fix")
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Test 1: DataFrame with ONLY 'primary_label' (reproducing user scenario)
    df_primary_only = pd.DataFrame([
        {
            "segment_id": "0001", "start": 0.0, "end": 6.0, "duration": 6.0,
            "primary_label": "speech", "cut_tier": "HIGH", "cut_certainty": 0.95,
            "cut_type": "voting_pause_midpoint", "needs_review": 0, "fallback_used": 0,
            "wav_file": "segments/segment_0001.wav"
        },
        {
            "segment_id": "0002", "start": 6.0, "end": 10.0, "duration": 4.0,
            "primary_label": "mixed", "cut_tier": "LOW", "cut_certainty": 0.35,
            "cut_type": "fallback_split", "needs_review": 1, "fallback_used": 1,
            "wav_file": "segments/segment_0002.wav"
        }
    ])
    
    try:
        ui = launch_segment_inspector(audio, df_primary_only, video_dir, sr=sr)
        print("[PASS] Test 1: launch_segment_inspector succeeded with primary_label only DataFrame!")
    except Exception as e:
        print("[FAIL] Test 1 FAILED:", e)
        raise e

    # Test 2: DataFrame with ONLY 'label' (legacy DataFrame)
    df_label_only = pd.DataFrame([
        {
            "segment_id": "0001", "start": 0.0, "end": 6.0, "duration": 6.0,
            "label": "music", "cut_tier": "HIGH", "cut_certainty": 0.95,
            "cut_type": "voting_pause_midpoint", "needs_review": 0, "fallback_used": 0,
            "wav_file": "segments/segment_0001.wav"
        }
    ])
    try:
        ui = launch_segment_inspector(audio, df_label_only, video_dir, sr=sr)
        print("[PASS] Test 2: launch_segment_inspector succeeded with label only DataFrame!")
    except Exception as e:
        print("[FAIL] Test 2 FAILED:", e)
        raise e

    # Test 3: sample_manual_check_window
    try:
        chk = sample_manual_check_window(df_primary_only, video_dir, minutes=0.1)
        assert "label" in chk.columns
        print("[PASS] Test 3: sample_manual_check_window succeeded and contains 'label' column!")
    except Exception as e:
        print("[FAIL] Test 3 FAILED:", e)
        raise e

    # Test 4: End-to-end cut_speech_and_mixed_segments into launch_segment_inspector
    regions_df = pd.DataFrame([
        {"region_id": "R001", "start": 0.0, "end": 10.0, "label": "speech", "confidence": 0.9}
    ])
    seg_df = cut_speech_and_mixed_segments(
        audio=audio, regions_df=regions_df, out_dir=video_dir, sr=sr,
        use_word_verification=False, run_ablation=False
    )
    assert "label" in seg_df.columns, "FAILURE: 'label' not in generated segments_df!"
    assert "primary_label" in seg_df.columns, "FAILURE: 'primary_label' not in generated segments_df!"
    print("[PASS] Test 4: cut_speech_and_mixed_segments generates both 'label' and 'primary_label'!")
    
    ui_e2e = launch_segment_inspector(audio, seg_df, video_dir, sr=sr)
    print("[PASS] Test 5: launch_segment_inspector runs cleanly on fresh engine output!")

    print("\n>>> ALL INSPECTOR TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_inspector_fixes()
