from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from .labelling import COLORS, DEFAULT_SR, LABELS


def plot_spectrogram_regions(y: np.ndarray, regions_df: pd.DataFrame,
                             gt_df: pd.DataFrame | None = None,
                             sr: int = DEFAULT_SR,
                             title: str = "Audio Spectrogram & Classification Regions",
                             fmax: int = 8000,
                             save_path: str | Path | None = None) -> plt.Figure:
    total_dur = len(y) / sr
    f, t, z = signal.stft(y, sr, nperseg=512, noverlap=384)
    s_db = 20.0 * np.log10(np.abs(z) + 1e-6)

    nrows = 3 if gt_df is not None else 2
    fig, axes = plt.subplots(
        nrows, 1, figsize=(15, 3.8 + 1.2 * (nrows - 1)), sharex=True,
        gridspec_kw={"height_ratios": [4] + [0.8] * (nrows - 1)}
    )

    if nrows == 2:
        ax_spec, ax_pred = axes[0], axes[1]
        ax_gt = None
    else:
        ax_spec, ax_pred, ax_gt = axes[0], axes[1], axes[2]

    vmax = s_db.max()
    vmin = vmax - 75.0
    ax_spec.pcolormesh(t, f, s_db, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax_spec.set_ylim(0, fmax)
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title(title, fontsize=11, fontweight="bold")

    # Min relative duration to print text to avoid squishing in long recordings
    min_rel_dur = 0.035 if total_dur > 120 else 0.015

    for r in regions_df.itertuples():
        color = COLORS.get(r.label, "#999999")
        conf = getattr(r, "confidence", 1.0)
        dur = r.end - r.start
        ax_pred.barh(0, dur, left=r.start, color=color, edgecolor="black", alpha=0.85, height=0.7)

        # Only draw text if region is wide enough to be readable
        if (dur / total_dur) >= min_rel_dur:
            ax_spec.axvline(r.start, color="white", lw=0.8, ls="--", alpha=0.5)
            ax_pred.text(
                (r.start + r.end) / 2.0, 0,
                f"{r.label}\n({conf:.2f})",
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color="white" if r.label != "silence" else "black"
            )

    ax_pred.set_yticks([])
    ax_pred.set_ylabel("Labels")

    if ax_gt is not None and gt_df is not None:
        for r in gt_df.itertuples():
            color = COLORS.get(r.label, "#999999")
            dur = r.end - r.start
            ax_gt.barh(0, dur, left=r.start, color=color, edgecolor="black", alpha=0.85, height=0.7)
            if (dur / total_dur) >= min_rel_dur:
                ax_gt.text(
                    (r.start + r.end) / 2.0, 0,
                    f"{r.label}",
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color="white" if r.label != "silence" else "black"
                )
        ax_gt.set_yticks([])
        ax_gt.set_ylabel("Truth")

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(0, total_dur)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_segments_overview(audio: np.ndarray, segments_df: pd.DataFrame,
                           sr: int = DEFAULT_SR, max_segments: int = 20,
                           fmax: int = 8000,
                           title: str | None = None,
                           save_path: str | Path | None = None) -> plt.Figure:
    """Plot high-resolution spectrogram of the first N segments alongside a rendered graphical metadata table.
    Eliminates text overlapping by zooming directly into the initial sample window.
    """
    if len(segments_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No segments available to plot", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    sub_df = segments_df.head(max_segments).copy().reset_index(drop=True)
    t_end = float(sub_df["end"].max())
    
    # Slice audio for the active window
    i_end = min(len(audio), int((t_end + 0.5) * sr))
    sliced_audio = audio[:i_end]

    f, t, z = signal.stft(sliced_audio, sr, nperseg=512, noverlap=384)
    s_db = 20.0 * np.log10(np.abs(z) + 1e-6)

    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.6, 1.0, 4.6], hspace=0.35)

    ax_spec = fig.add_subplot(gs[0])
    ax_bars = fig.add_subplot(gs[1], sharex=ax_spec)
    ax_tbl = fig.add_subplot(gs[2])

    vmax = s_db.max()
    vmin = vmax - 75.0
    ax_spec.pcolormesh(t, f, s_db, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax_spec.set_ylim(0, fmax)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=10, fontweight="bold")
    
    if title is None:
        title = f"Spectrogram & Pause Cuts (First {len(sub_df)} Segments | 5–15s)"
    ax_spec.set_title(title, fontsize=13, fontweight="bold", pad=8)

    color_map = {"speech": "#1f77b4", "mixed": "#ff7f0e"}

    for row in sub_df.itertuples():
        seg_color = color_map.get(row.source_region_label, "#2ca02c")
        
        # Segment vertical boundary lines
        ax_spec.axvline(row.start, color="cyan", lw=1.2, ls="--", alpha=0.85)
        ax_spec.axvline(row.end, color="yellow", lw=1.2, ls=":", alpha=0.85)

        # Timeline bar
        width = row.end - row.start
        ax_bars.barh(0, width, left=row.start, color=seg_color, edgecolor="black", alpha=0.85, height=0.7)
        
        # Label text inside segment bar
        ax_bars.text(
            row.start + width / 2.0, 0,
            f"#{row.segment_id}\n{row.source_region_label}\n{row.duration:.1f}s",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="white"
        )

    ax_bars.set_yticks([])
    ax_bars.set_ylabel("Segments", fontsize=10, fontweight="bold")
    ax_bars.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
    ax_bars.set_xlim(0, t_end + 0.5)

    # Graphical Metadata Table Subplot
    ax_tbl.axis("off")
    ax_tbl.set_title(f"Segment Index Table (Samples 1 to {len(sub_df)})", fontsize=12, fontweight="bold", pad=12)

    table_cols = ["Segment ID", "Start (s)", "End (s)", "Duration (s)", "Label", "Confidence", "Audio File"]
    table_data = []
    for r in sub_df.itertuples():
        table_data.append([
            r.segment_id,
            f"{r.start:.2f}",
            f"{r.end:.2f}",
            f"{r.duration:.2f}",
            r.source_region_label,
            f"{r.confidence:.2f}",
            r.wav_file
        ])

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=table_cols,
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.25)

    # Style table header and alternating row colors
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#1f4e78")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if row_idx % 2 == 1:
                cell.set_facecolor("#f2f5f9")
            else:
                cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#d0d0d0")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved visual report image: {save_path}")

    return fig


class RegionEditor:
    """Manual adjustment tool for region boundaries and labels."""

    def __init__(self, regions_df: pd.DataFrame):
        self.regions = regions_df.copy().reset_index(drop=True)
        self.edit_log: list[dict] = []

    def _log(self, action: str, details: str, by: str):
        self.edit_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
            "modified_by": by,
        })

    def move_boundary(self, index: int, new_time: float, by: str = "annotator") -> pd.DataFrame:
        r = self.regions
        if index < 0 or index >= len(r) - 1:
            raise IndexError("Index out of bounds.")

        left_start = r.loc[index, "start"]
        right_end = r.loc[index + 1, "end"]
        if not (left_start < new_time < right_end):
            raise ValueError(f"Boundary must stay within ({left_start:.2f}s, {right_end:.2f}s).")

        old_time = r.loc[index, "end"]
        r.loc[index, "end"] = round(new_time, 3)
        r.loc[index, "duration"] = round(r.loc[index, "end"] - r.loc[index, "start"], 3)
        r.loc[index + 1, "start"] = round(new_time, 3)
        r.loc[index + 1, "duration"] = round(r.loc[index + 1, "end"] - r.loc[index + 1, "start"], 3)
        r.loc[[index, index + 1], "confidence"] = 1.0

        self._log("move_boundary", f"Index {index}: {old_time:.2f}s -> {new_time:.2f}s", by)
        return self.regions

    def relabel(self, index: int, new_label: str, by: str = "annotator") -> pd.DataFrame:
        if new_label not in LABELS:
            raise ValueError(f"Invalid label '{new_label}'.")

        old_label = self.regions.loc[index, "label"]
        self.regions.loc[index, "label"] = new_label
        self.regions.loc[index, "confidence"] = 1.0
        self._log("relabel", f"Region {index}: '{old_label}' -> '{new_label}'", by)
        return self.regions

    def split_region(self, index: int, split_time: float, by: str = "annotator") -> pd.DataFrame:
        r = self.regions
        row = r.loc[index].copy()
        if not (row.start < split_time < row.end):
            raise ValueError(f"Split time must be within ({row.start:.2f}s, {row.end:.2f}s).")

        a, b = row.copy(), row.copy()
        a["end"] = round(split_time, 3)
        a["duration"] = round(a["end"] - a["start"], 3)
        a["confidence"] = 1.0

        b["start"] = round(split_time, 3)
        b["duration"] = round(b["end"] - b["start"], 3)
        b["confidence"] = 1.0

        self.regions = pd.concat([r.loc[:index - 1], pd.DataFrame([a, b]), r.loc[index + 1:]]).reset_index(drop=True)
        self._log("split_region", f"Region {index} split at {split_time:.2f}s", by)
        return self.regions

    def merge_with_next(self, index: int, by: str = "annotator") -> pd.DataFrame:
        r = self.regions
        if index < 0 or index >= len(r) - 1:
            raise IndexError("Cannot merge last region.")

        self.regions.loc[index, "end"] = r.loc[index + 1, "end"]
        self.regions.loc[index, "duration"] = round(self.regions.loc[index, "end"] - self.regions.loc[index, "start"], 3)
        self.regions.loc[index, "confidence"] = 1.0
        self.regions = self.regions.drop(index=index + 1).reset_index(drop=True)

        self._log("merge_with_next", f"Regions {index} and {index + 1} merged", by)
        return self.regions

    def get_edits_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.edit_log)

    def save(self, regions_csv_path: str | Path, edits_csv_path: str | Path | None = None) -> pd.DataFrame:
        regions_csv_path = Path(regions_csv_path)
        regions_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.regions.to_csv(regions_csv_path, index=False)

        if edits_csv_path is None:
            edits_csv_path = regions_csv_path.parent / f"{regions_csv_path.stem}_edits.csv"
        else:
            edits_csv_path = Path(edits_csv_path)
        
        self.get_edits_df().to_csv(edits_csv_path, index=False)
        return self.regions
