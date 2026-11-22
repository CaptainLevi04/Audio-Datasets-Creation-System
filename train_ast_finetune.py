#!/usr/bin/env python3
"""Phase 6: Audio Spectrogram Transformer (AST) Fine-Tuning & Evaluation Script.
Implements:
1. AST Multi-Task Dataset (5s windows @ 16kHz with dual labels: Content & Quality).
2. Step A: Linear Probe (Train heads only, lr=1e-3, 3 epochs).
3. Step B: Full Fine-Tune (Backbone + heads, lr=2e-5, 3-5 epochs, fp16).
4. Validation Macro-F1 checkpoint selection.
5. Evaluation on Frozen Test Set (Confusion Matrix, Precision/Recall/F1, Per-class breakdown).
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from speech_corpus.ast_model import (
    ASTMultiTaskClassifier,
    CONTENT_LABELS,
    QUALITY_LABELS,
    find_model_weights_dir,
    get_default_device,
)
from speech_corpus.audio import DEFAULT_SR, read_wav


class ASTMultiTaskDataset(Dataset):
    """PyTorch Dataset loading 5s WAV segments or slicing from audio for AST training."""

    def __init__(self, df: pd.DataFrame, feature_extractor, base_dir: Path, win_s: float = 5.0):
        self.df = df.reset_index(drop=True)
        self.fe = feature_extractor
        self.base_dir = Path(base_dir)
        self.win_samples = int(win_s * DEFAULT_SR)

        self.c2i = {lbl: i for i, lbl in enumerate(CONTENT_LABELS)}
        self.q2i = {lbl: i for i, lbl in enumerate(QUALITY_LABELS)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = self.base_dir / row["wav_file"]
        
        try:
            samples, sr = read_wav(wav_path)
        except Exception:
            # Fallback zero array if missing
            samples = np.zeros(self.win_samples, dtype=np.float32)
            sr = DEFAULT_SR

        # Pad or trim to exactly win_samples
        if len(samples) < self.win_samples:
            samples = np.pad(samples, (0, self.win_samples - len(samples)))
        else:
            samples = samples[:self.win_samples]

        # Extract features (1024 x 128)
        feats = self.fe(samples, sampling_rate=sr, return_tensors="pt")
        input_values = feats.input_values.squeeze(0)

        # Labels
        content_str = str(row.get("human_content") or row.get("source_region_label") or "speech").lower()
        content_label = self.c2i.get(content_str, 0)

        quality_str = str(row.get("human_quality") or row.get("pred_quality") or "clean").lower()
        quality_label = self.q2i.get(quality_str, 0)

        return {
            "input_values": input_values,
            "content_label": torch.tensor(content_label, dtype=torch.long),
            "quality_label": torch.tensor(quality_label, dtype=torch.long),
        }


def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, Any]:
    """Evaluate AST on validation/test set and compute Macro-F1 & Confusion Matrices."""
    model.eval()
    all_c_true, all_c_pred = [], []
    all_q_true, all_q_pred = [], []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_values"].to(device)
            c_logits, q_logits = model(inputs)

            c_preds = c_logits.argmax(dim=-1).cpu().numpy()
            q_preds = q_logits.argmax(dim=-1).cpu().numpy()

            all_c_true.extend(batch["content_label"].numpy())
            all_c_pred.extend(c_preds)

            all_q_true.extend(batch["quality_label"].numpy())
            all_q_pred.extend(q_preds)

    c_macro_f1 = f1_score(all_c_true, all_c_pred, average="macro", zero_division=0)
    q_f1 = f1_score(all_q_true, all_q_pred, average="macro", zero_division=0)

    c_cm = confusion_matrix(all_c_true, all_c_pred, labels=list(range(4)))
    c_report = classification_report(all_c_true, all_c_pred, target_names=CONTENT_LABELS, zero_division=0, output_dict=True)

    return {
        "content_macro_f1": round(float(c_macro_f1), 4),
        "quality_f1": round(float(q_f1), 4),
        "content_cm": c_cm,
        "content_report": c_report,
        "c_true": all_c_true,
        "c_pred": all_c_pred,
    }


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: str, scaler: torch.amp.GradScaler | None = None) -> float:
    """Train single epoch with mixed precision (fp16)."""
    model.train()
    criterion_c = nn.CrossEntropyLoss()
    criterion_q = nn.CrossEntropyLoss()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        inputs = batch["input_values"].to(device)
        c_targets = batch["content_label"].to(device)
        q_targets = batch["quality_label"].to(device)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            c_logits, q_logits = model(inputs)
            loss_c = criterion_c(c_logits, c_targets)
            loss_q = criterion_q(q_logits, q_targets)
            loss = loss_c + 0.5 * loss_q

        if scaler and device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser(description="AST Multi-Task Fine-Tuning (Phase 6)")
    parser.add_argument("--splits-dir", type=str, default="splits", help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--data-dir", type=str, default="data", help="Root directory for audio clips")
    parser.add_argument("--output-dir", type=str, default="models/ast_finetuned", help="Save directory for model weights")
    parser.add_argument("--epochs-linear", type=int, default=3, help="Step A Linear probe epochs (default: 3)")
    parser.add_argument("--epochs-full", type=int, default=4, help="Step B Full fine-tune epochs (default: 4)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()
    device = get_default_device() if args.device == "auto" else args.device
    print(f"Executing AST Fine-Tuning on: {device.upper()}")

    splits_dir = Path(args.splits_dir)
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        print(f"Error: Split CSV files not found in {splits_dir}. Run Phase 5 data split first.")
        return

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Loaded splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Load feature extractor & base model weights
    from transformers import AutoFeatureExtractor
    weights_path = find_model_weights_dir()
    feature_extractor = AutoFeatureExtractor.from_pretrained(weights_path)

    base_dir = Path(args.data_dir)
    train_ds = ASTMultiTaskDataset(train_df, feature_extractor, base_dir=base_dir)
    val_ds = ASTMultiTaskDataset(val_df, feature_extractor, base_dir=base_dir)
    test_ds = ASTMultiTaskDataset(test_df, feature_extractor, base_dir=base_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Step A: Linear Probe (Freeze backbone, lr=1e-3)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP A: Linear Probe (Training new heads only, lr=1e-3)")
    print("=" * 65)
    model = ASTMultiTaskClassifier(weights_path, freeze_backbone=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val_f1 = -1.0
    best_weights_path = out_dir / "ast_best_checkpoint.pt"

    for epoch in range(1, args.epochs_linear + 1):
        loss = train_epoch(model, train_loader, optimizer, device, scaler=scaler)
        val_res = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs_linear} | Loss: {loss:.4f} | Val Content Macro-F1: {val_res['content_macro_f1']} | Quality F1: {val_res['quality_f1']}")

        if val_res["content_macro_f1"] > best_val_f1:
            best_val_f1 = val_res["content_macro_f1"]
            torch.save(model.state_dict(), best_weights_path)

    # -------------------------------------------------------------
    # Step B: Full Fine-Tune (Unfreeze backbone, lr=2e-5, fp16)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP B: Full Fine-Tune (Backbone + heads, lr=2e-5, fp16)")
    print("=" * 65)
    model.load_state_dict(torch.load(best_weights_path))
    for p in model.ast.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": model.ast.parameters(), "lr": 2e-5},
        {"params": model.content_head.parameters(), "lr": 1e-4},
        {"params": model.quality_head.parameters(), "lr": 1e-4},
    ], weight_decay=1e-2)

    for epoch in range(1, args.epochs_full + 1):
        loss = train_epoch(model, train_loader, optimizer, device, scaler=scaler)
        val_res = evaluate_model(model, val_loader, device)
        print(f"Full FT Epoch {epoch}/{args.epochs_full} | Loss: {loss:.4f} | Val Content Macro-F1: {val_res['content_macro_f1']} | Quality F1: {val_res['quality_f1']}")

        if val_res["content_macro_f1"] > best_val_f1:
            best_val_f1 = val_res["content_macro_f1"]
            torch.save(model.state_dict(), best_weights_path)

    # -------------------------------------------------------------
    # Final Evaluation on Frozen Test Set
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL EVALUATION ON FROZEN TEST SET")
    print("=" * 65)
    model.load_state_dict(torch.load(best_weights_path))
    test_res = evaluate_model(model, test_loader, device)

    print(f"\nTest Macro-F1 (Content): {test_res['content_macro_f1']}")
    print(f"Test Macro-F1 (Quality): {test_res['quality_f1']}")
    print("\nPer-Class Classification Report:")
    for cls_name in CONTENT_LABELS:
        s = test_res["content_report"].get(cls_name, {})
        print(f"  {cls_name.upper():<10} | Precision: {s.get('precision', 0):.3f} | Recall: {s.get('recall', 0):.3f} | F1: {s.get('f1-score', 0):.3f}")

    print("\nConfusion Matrix:")
    print(pd.DataFrame(test_res["content_cm"], index=CONTENT_LABELS, columns=CONTENT_LABELS))

    # Save results table
    results_summary = pd.DataFrame([{
        "model": "AST_FineTuned",
        "macro_f1_content": test_res["content_macro_f1"],
        "macro_f1_quality": test_res["quality_f1"],
        "speech_f1": test_res["content_report"].get("speech", {}).get("f1-score", 0),
        "music_f1": test_res["content_report"].get("music", {}).get("f1-score", 0),
        "mixed_f1": test_res["content_report"].get("mixed", {}).get("f1-score", 0),
        "silence_f1": test_res["content_report"].get("silence", {}).get("f1-score", 0),
    }])
    results_path = out_dir / "results_test.csv"
    results_summary.to_csv(results_path, index=False)
    print(f"\nFinal test evaluation saved to: {results_path}")


if __name__ == "__main__":
    main()
