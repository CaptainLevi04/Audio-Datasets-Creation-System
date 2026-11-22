"""Interactive UI Inspectors for Speech Corpus Pipeline (Colab & Local Jupyter).
Includes:
1. Spectrogram + Region Editor (0-8 kHz spectrogram, move borders / relabel / split / merge / delete / add).
2. 60 FPS Real-Time Moving Pointer across Waveform and Energy Curve (in synchronized HTML5 iframe).
3. VAD Pauses and Safe-Zone Visualizer.
4. Segment Inspector & Review Queue (relabel, mark reviewed, save & audit log).
5. 10-Minute Manual Check Tool (ground truth verification required by brief).
"""

from pathlib import Path
import io
import json
import base64
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from IPython.display import display, Audio, HTML, clear_output
import ipywidgets as widgets

from .audio import DEFAULT_SR
from .segmentation import log_human_change
from .ast_model import (
    region_set_label, region_move_borders, region_split,
    region_merge_with_next, region_delete, region_paint
)

LABEL_COLORS = {"speech": "#2ca02c", "music": "#1f77b4", "mixed": "#ff7f0e", "silence": "#7f7f7f"}


def plot_spectrogram_with_regions(audio: np.ndarray, regions_df: pd.DataFrame,
                                  t0: float, t1: float, sr: int = DEFAULT_SR,
                                  selected: Optional[int] = None, title: str = ""):
    """Linear-frequency spectrogram (0-8 kHz, 10 ms hop) with coloured region overlays and region borders."""
    t0, t1 = max(0.0, t0), min(len(audio) / sr, t1)
    seg = audio[int(t0 * sr):int(t1 * sr)]
    f, tt, S = signal.spectrogram(seg, fs=sr, nperseg=400, noverlap=240, nfft=512, mode="psd")
    S_db = 10 * np.log10(S + 1e-12)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.imshow(S_db, origin="lower", aspect="auto", extent=[t0, t0 + S_db.shape[1] * 0.010, 0, sr / 2000.0],
              vmin=S_db.max() - 80, vmax=S_db.max(), cmap="magma")
    for i, r in regions_df.iterrows():
        if r["end"] < t0 or r["start"] > t1:
            continue
        a, b = max(t0, r["start"]), min(t1, r["end"])
        ax.axvspan(a, b, color=LABEL_COLORS.get(r["label"], "w"), alpha=0.28 if i != selected else 0.45)
        ax.axvline(r["start"], color="white", ls="--", lw=1.0)
        ax.axvline(r["end"], color="white", ls="--", lw=1.0)
        ax.text((a + b) / 2, sr / 2000.0 * 0.93, f"#{i} {r['label']}", color="white", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title or "Spectrogram + regions")
    plt.tight_layout()
    return fig


def build_synchronized_player_iframe(audio_samples: np.ndarray, sr: int, t_val: float, win_dur: float,
                                     intervals_df: pd.DataFrame, times_arr: np.ndarray,
                                     energy_db_arr: np.ndarray, thresh_db: float, total_audio_dur: float) -> str:
    """Render self-contained HTML5 Audio Player with synchronized moving cursor inside an iframe."""
    import html
    win_s = max(0.0, t_val - (win_dur / 2.0))
    win_e = min(total_audio_dur, win_s + win_dur)
    if (win_e - win_s) < win_dur and win_s > 0:
        win_s = max(0.0, win_e - win_dur)
    snippet = audio_samples[int(win_s * sr):int(win_e * sr)]

    fig, (ax_wave, ax_energy) = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.0]})
    t_ax = np.linspace(win_s, win_e, len(snippet))
    ax_wave.plot(t_ax, snippet, color="#1f77b4", linewidth=0.8)
    sub_runs = intervals_df[(intervals_df["end"] >= win_s) & (intervals_df["start"] <= win_e)]
    for _, r in sub_runs.iterrows():
        ax_wave.axvspan(max(win_s, r["start"]), min(win_e, r["end"]),
                        color="#d4edda" if r["state"] == "SPEECH" else "#e2e3e5", alpha=0.45)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title("Voice Activity Overlay • synchronized cursor", fontsize=10, fontweight="bold")
    ax_wave.grid(True, linestyle=":", alpha=0.4)

    k0, k1 = max(0, int(win_s / 0.010)), min(len(times_arr), int(win_e / 0.010))
    sub_e = energy_db_arr[k0:k1]
    if k1 > k0:
        ax_energy.plot(times_arr[k0:k1], sub_e, color="#2ca02c", linewidth=1.1, label="Short-Time Energy (dB)")
    ax_energy.axhline(thresh_db, color="#e0a800", linestyle=":", linewidth=1.6, label=f"Silence threshold ({thresh_db:.1f} dB)")
    ax_energy.set_ylabel("Energy (dB)")
    ax_energy.set_xlabel("Time (seconds)")
    ax_energy.set_xlim(win_s, win_e)
    ax_energy.legend(loc="lower right", fontsize=8)
    ax_energy.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    pos_w, pos_e = ax_wave.get_position(), ax_energy.get_position()
    x0, x1 = float(pos_w.x0), float(pos_w.x1)
    y_top, y_bot = float(1.0 - pos_w.y1), float(1.0 - pos_e.y0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

    abuf = io.BytesIO()
    sf.write(abuf, snippet, sr, format="WAV", subtype="PCM_16")
    b64_aud = base64.b64encode(abuf.getvalue()).decode("ascii")

    intvs_json = json.dumps([{
        "start": round(float(r["start"]), 3), "end": round(float(r["end"]), 3),
        "duration": round(float(r["duration"]), 3), "state": str(r["state"])
    } for _, r in sub_runs.iterrows()])
    energy_json = json.dumps(np.round(np.clip(sub_e, -65.0, 0.0), 1).tolist())

    inner_html = f'''<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body {{ margin:0; padding:6px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:#fafafa; }}
.badge {{ padding:6px 12px; border-radius:4px; font-weight:600; font-size:13px; margin-bottom:6px; display:inline-block; }}
.badge-speech {{ background:#d4edda; color:#155724; border:1px solid #c3e6cb; }}
.badge-silence {{ background:#e2e3e5; color:#383d41; border:1px solid #d6d8db; }}
.plot-box {{ position:relative; display:inline-block; width:100%; max-width:820px; }}
.plot-box img {{ width:100%; display:block; }}
.cursor {{ position:absolute; top:{y_top * 100:.2f}%; height:{(y_bot - y_top) * 100:.2f}%; width:2px; background:#dc3545; box-shadow:0 0 4px #dc3545; pointer-events:none; display:none; }}
audio {{ width:100%; max-width:820px; margin-top:6px; }}
</style></head><body>
<div id="badge" class="badge badge-silence">Initializing...</div>
<div class="plot-box" id="pbox">
  <img src="data:image/png;base64,{b64_img}" id="img">
  <div class="cursor" id="cur"></div>
</div>
<audio id="aud" controls src="data:audio/wav;base64,{b64_aud}"></audio>
<script>
var win_s = {win_s:.3f}, win_e = {win_e:.3f}, dur = win_e - win_s;
var x0 = {x0:.4f}, x1 = {x1:.4f};
var intervals = {intvs_json};
var energy = {energy_json};
var thresh = {thresh_db:.1f};
var aud = document.getElementById("aud"), cur = document.getElementById("cur"), badge = document.getElementById("badge"), img = document.getElementById("img");

function getState(t) {{
  for (var i = 0; i < intervals.length; i++) {{
    if (t >= intervals[i].start && t <= intervals[i].end) return intervals[i].state;
  }}
  return "SILENCE";
}}
function upd() {{
  var t_rel = aud.currentTime;
  var t_abs = win_s + t_rel;
  var frac = Math.max(0, Math.min(1, t_rel / dur));
  var w = img.clientWidth;
  var px = (x0 + frac * (x1 - x0)) * w;
  cur.style.left = px + "px";
  cur.style.display = "block";
  var st = getState(t_abs);
  if (st === "SPEECH") {{
    badge.className = "badge badge-speech";
    badge.innerHTML = "🎙️ TIME " + t_abs.toFixed(2) + "s → NO SILENCE [ACTIVE SPEECH]";
  }} else {{
    badge.className = "badge badge-silence";
    badge.innerHTML = "⏸️ TIME " + t_abs.toFixed(2) + "s → SILENCE [NO SPEECH / PAUSE]";
  }}
}}
aud.ontimeupdate = upd;
aud.onseeked = upd;
aud.onplay = function() {{ cur.style.display = "block"; }};
img.onclick = function(e) {{
  var rect = img.getBoundingClientRect();
  var frac_x = (e.clientX - rect.left) / rect.width;
  var frac = (frac_x - x0) / (x1 - x0);
  if (frac >= 0 && frac <= 1) {{
    aud.currentTime = frac * dur;
    aud.play();
  }}
}};
upd();
</script></body></html>'''
    esc = html.escape(inner_html)
    return f'<iframe srcdoc="{esc}" style="width:100%; height:490px; border:none; overflow:hidden;" scrolling="no"></iframe>'


def launch_spectrogram_region_editor(audio: np.ndarray, regions_df: pd.DataFrame,
                                     video_dir: str | Path, sr: int = DEFAULT_SR,
                                     user: str = "annotator"):
    """Spectrogram editor: select a region, listen, move start/end borders, relabel, split, merge, delete, add."""
    video_dir = Path(video_dir)
    reg_csv = video_dir / "regions.csv"
    state = {"df": regions_df.copy()}

    def _save_and_log(action, idx, old_v, new_v, reason=""):
        state["df"].to_csv(reg_csv, index=False)
        log_human_change(video_dir, user, idx, action, old_v, new_v, reason)

    sel = widgets.Dropdown(description="Region:", layout=widgets.Layout(width="70%"))
    play_out = widgets.Output()
    spec_out = widgets.Output()

    lbl_drop = widgets.Dropdown(options=["speech", "music", "mixed", "silence"], description="Label:")
    b_relabel = widgets.Button(description="Set Label", button_style="warning")

    w_start = widgets.FloatText(description="Start (s):", step=0.1)
    w_end = widgets.FloatText(description="End (s):", step=0.1)
    b_move = widgets.Button(description="Move Borders", button_style="info")

    w_split = widgets.FloatText(description="Split at (s):", step=0.1)
    b_split = widgets.Button(description="Split Region", button_style="primary")

    b_merge = widgets.Button(description="Merge with Next", button_style="danger")
    b_del = widgets.Button(description="Delete Region", button_style="danger")

    def _sync_opts():
        sel.options = [(f"#{i:02d} [{r.label}] {r.start:.2f}-{r.end:.2f}s (dur {r.duration:.2f}s, urg {r.urgency:.2f}) {'✔' if r.status=='human_edited' else ''}", i)
                       for i, r in state["df"].iterrows()]

    def redraw(_=None):
        _sync_opts()
        i = sel.value if sel.value is not None else 0
        r = state["df"].loc[i]
        lbl_drop.value = r.label
        w_start.value, w_end.value, w_split.value = float(r.start), float(r.end), round((r.start + r.end) / 2, 2)
        play_out.clear_output()
        with play_out:
            display(Audio(audio[int(r.start * sr):int(r.end * sr)], rate=sr))
        spec_out.clear_output()
        with spec_out:
            fig = plot_spectrogram_with_regions(
                audio, state["df"], max(0, r.start - 5), min(len(audio)/sr, r.end + 5),
                sr, selected=i, title=f"Region #{i} ({r.label}) {r.start:.2f}-{r.end:.2f}s"
            )
            display(fig)
            plt.close(fig)

    def on_relabel(_):
        i = sel.value
        old = state["df"].loc[i, "label"]
        state["df"] = region_set_label(state["df"], i, lbl_drop.value)
        _save_and_log("relabel", i, old, lbl_drop.value)
        redraw()

    def on_move(_):
        i = sel.value
        old = f"{state['df'].loc[i, 'start']}-{state['df'].loc[i, 'end']}"
        state["df"] = region_move_borders(state["df"], i, w_start.value, w_end.value)
        _save_and_log("move_borders", i, old, f"{w_start.value}-{w_end.value}")
        redraw()

    def on_split(_):
        i = sel.value
        state["df"] = region_split(state["df"], i, w_split.value)
        _save_and_log("split", i, "-", f"split at {w_split.value}s")
        redraw()

    def on_merge(_):
        i = sel.value
        state["df"] = region_merge_with_next(state["df"], i)
        _save_and_log("merge_with_next", i, "-", "merged")
        redraw()

    def on_delete(_):
        i = sel.value
        state["df"] = region_delete(state["df"], i)
        _save_and_log("delete", i, "-", "deleted")
        redraw()

    b_relabel.on_click(on_relabel)
    b_move.on_click(on_move)
    b_split.on_click(on_split)
    b_merge.on_click(on_merge)
    b_del.on_click(on_delete)
    sel.observe(redraw, names="value")

    ui = widgets.VBox([
        widgets.HTML("<h3>🔍 Inspector 1: Spectrogram & Macro-Region Editor</h3>"),
        sel,
        play_out,
        widgets.HBox([lbl_drop, b_relabel]),
        widgets.HBox([w_start, w_end, b_move]),
        widgets.HBox([w_split, b_split, b_merge, b_del]),
        spec_out
    ])
    display(ui)
    redraw()
    return ui


def launch_timeline_pointer_inspector(audio: np.ndarray, pauses_df: pd.DataFrame,
                                      vad_timeline_df: pd.DataFrame, sr: int = DEFAULT_SR):
    """Moving pointer inspector with real-time audio playback synchronization."""
    from .segmentation import compute_instantaneous_energy_vad_overlay
    intvs, times, energy, thresh = compute_instantaneous_energy_vad_overlay(audio, sr, vad_timeline_df)
    tot = len(audio) / sr

    sl = widgets.FloatSlider(value=min(10.0, tot / 2), min=0, max=max(tot - 1, 1), step=1.0,
                             description="Center (s):", layout=widgets.Layout(width="60%"))
    out = widgets.Output()

    def show(_=None):
        out.clear_output()
        with out:
            html = build_synchronized_player_iframe(audio, sr, sl.value, 15.0, intvs, times, energy, thresh, tot)
            display(HTML(html))

    sl.observe(show, names="value")
    display(widgets.VBox([
        widgets.HTML("<h3>📍 Inspector 2: 60 FPS Moving Pointer & Voice Activity Overlay</h3>"),
        sl, out
    ]))
    show()


def launch_vad_overlay_inspector(audio: np.ndarray, pauses_df: pd.DataFrame,
                                 vad_timeline_df: pd.DataFrame, sr: int = DEFAULT_SR):
    """Pause & boundary safe zones inspector."""
    if pauses_df.empty:
        print("No pauses to inspect.")
        return
    tot = len(audio) / sr
    sel = widgets.Dropdown(options=[(f"{r.pause_id}: {r.start:.2f}-{r.end:.2f}s ({r.duration_ms:.0f}ms) {'✅ SAFE' if r.is_safe else '⚠️ SHORT'}", i)
                                    for i, r in pauses_df.iterrows()], description="Pause:", layout=widgets.Layout(width="60%"))
    out = widgets.Output()

    def show(_=None):
        out.clear_output()
        r = pauses_df.loc[sel.value]
        fig, ax = plt.subplots(figsize=(10, 2.5))
        t0, t1 = max(0, r.start - 2.0), min(tot, r.end + 2.0)
        seg = audio[int(t0 * sr):int(t1 * sr)]
        ax.plot(np.linspace(t0, t1, len(seg)), seg, color="#1f77b4", lw=0.8)
        ax.axvspan(r.start, r.end, color="#ffc107" if not r.is_safe else "#28a745", alpha=0.35, label=f"Pause ({r.duration_ms:.0f}ms)")
        ax.axvline(r.center, color="#dc3545", ls="--", label="Midpoint cut")
        ax.set_xlim(t0, t1)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        with out:
            display(fig)
            plt.close(fig)
            display(Audio(seg, rate=sr))

    sel.observe(show, names="value")
    display(widgets.VBox([
        widgets.HTML("<h3>🔍 Inspector 3: Pause & Safe-Zone Inspector</h3>"),
        sel, out
    ]))
    show()


def launch_segment_inspector(audio: np.ndarray, segments_df: pd.DataFrame,
                             video_dir: str | Path, sr: int = DEFAULT_SR,
                             user: str = "annotator"):
    """Segment review queue inspector with audio playback, relabeling, and audit logging."""
    video_dir = Path(video_dir)
    seg_csv = video_dir / "segments.csv"
    
    df_copy = segments_df.copy()
    if "label" not in df_copy.columns and "primary_label" in df_copy.columns:
        df_copy["label"] = df_copy["primary_label"]
    elif "primary_label" not in df_copy.columns and "label" in df_copy.columns:
        df_copy["primary_label"] = df_copy["label"]
    state = {"df": df_copy}

    f_drop = widgets.Dropdown(
        options=[("ALL segments", "all"), ("LOW certainty only", "low"),
                 ("Fallback used only", "fallback"), ("NEEDS REVIEW only", "review")],
        description="Filter:"
    )
    sel = widgets.Dropdown(description="Segment:", layout=widgets.Layout(width="70%"))
    lbl_drop = widgets.Dropdown(options=["speech", "music", "mixed", "silence"], description="Label:")
    b_save = widgets.Button(description="💾 Save Relabel", button_style="success")
    b_rev = widgets.Button(description="✔ Mark Reviewed", button_style="info")
    out = widgets.Output()

    def _sync():
        f = f_drop.value
        df = state["df"]
        if f == "low":
            sub = df[df["cut_tier"] == "LOW"]
        elif f == "fallback":
            sub = df[df["fallback_used"] == 1]
        elif f == "review":
            sub = df[df["needs_review"] == 1]
        else:
            sub = df
        sel.options = [(f"{r.segment_id} [{r.cut_tier}] {r.label} ({r.duration:.2f}s) {'⚠️ REVIEW' if r.needs_review else ''}", i)
                       for i, r in sub.iterrows()]

    def redraw(_=None):
        _sync()
        out.clear_output()
        if sel.value is None or sel.value not in state["df"].index:
            return
        r = state["df"].loc[sel.value]
        lbl_drop.value = r.label if r.label in ["speech", "music", "mixed", "silence"] else "speech"
        with out:
            print(f"Segment: {r.segment_id} | Label: {r.label} | Duration: {r.duration:.2f}s | Certainty: {r.cut_certainty} ({r.cut_tier})")
            display(Audio(audio[int(r.start * sr):int(r.end * sr)], rate=sr))

    def on_save(_):
        i = sel.value
        old = state["df"].loc[i, "label"]
        state["df"].loc[i, "label"] = lbl_drop.value
        if "primary_label" in state["df"].columns:
            state["df"].loc[i, "primary_label"] = lbl_drop.value
        state["df"].loc[i, "needs_review"] = 0
        state["df"].to_csv(seg_csv, index=False)
        log_human_change(video_dir, user, state["df"].loc[i, "segment_id"], "relabel", old, lbl_drop.value)
        redraw()

    def on_rev(_):
        i = sel.value
        state["df"].loc[i, "needs_review"] = 0
        state["df"].to_csv(seg_csv, index=False)
        log_human_change(video_dir, user, state["df"].loc[i, "segment_id"], "mark_reviewed", 1, 0)
        redraw()

    f_drop.observe(redraw, names="value")
    sel.observe(redraw, names="value")
    b_save.on_click(on_save)
    b_rev.on_click(on_rev)

    ui = widgets.VBox([
        widgets.HTML("<h3>🔍 Inspector 4: Cut Segments & Human Review Queue</h3>"),
        widgets.HBox([f_drop, sel]),
        widgets.HBox([lbl_drop, b_save, b_rev]),
        out
    ])
    display(ui)
    redraw()
    return ui


# ---- Manual ground-truth check (Full Video & Review Queue) ----
def sample_manual_check_window(segments_df: pd.DataFrame, video_dir: str | Path,
                               minutes: Optional[float] = None, seed: int = 0) -> pd.DataFrame:
    """Prepare manual check DataFrame. If minutes is None or <= 0, includes the FULL VIDEO (all segments)."""
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    csv_path = video_dir / "manual_check.csv"
    
    d = segments_df.sort_values("start").reset_index(drop=True).copy()
    if "label" not in d.columns and "primary_label" in d.columns:
        d["label"] = d["primary_label"]
    elif "primary_label" not in d.columns and "label" in d.columns:
        d["primary_label"] = d["label"]

    def _is_flagged(r):
        seg_flag = (
            int(r.get("needs_review", 0)) == 1 or
            str(r.get("review_priority", "LOW")).upper() in ["URGENT", "MEDIUM"] or
            str(r.get("cut_tier", "")).upper() == "LOW" or
            int(r.get("fallback_used", 0)) == 1 or
            str(r.get("word_check_status", "")) == "WORD_SPLIT_RISK" or
            (str(r.get("review_reason", "none")).lower() not in ["none", "", "nan"]) or
            int(r.get("margin_ok", 1)) == 0 or
            str(r.get("decision_class", "")).upper() in ["WEAK", "FALLBACK_SPLIT", "BLIND_CUT"]
        )
        conf = float(r.get("confidence", 1.0)) if pd.notnull(r.get("confidence")) else 1.0
        lbl = str(r.get("primary_label", r.get("label", "speech"))).lower()
        lbl_flag = (conf < 0.70 or (lbl == "mixed" and conf < 0.85))
        return bool(seg_flag or lbl_flag)

    d["is_flagged"] = d.apply(_is_flagged, axis=1)

    if minutes is not None and minutes > 0 and d["duration"].sum() > minutes * 60:
        rng = np.random.default_rng(seed)
        tot = d.duration.sum()
        for _ in range(50):
            i = int(rng.integers(0, len(d)))
            cum = d.duration.iloc[i:].cumsum()
            if cum.iloc[-1] >= minutes * 60 or tot < minutes * 60:
                break
        j = i + int((cum <= minutes * 60).sum()) + 1
        chk = d.iloc[i:j].copy().reset_index(drop=True)
        print(f"Manual-check sample: {len(chk)} segments, {chk.duration.sum() / 60:.1f} min -> manual_check.csv")
    else:
        chk = d.copy()
        print(f"Manual-check full video: {len(chk)} segments, {chk.duration.sum() / 60:.1f} min -> manual_check.csv")

    # Preserve previous progress if manual_check.csv already exists
    if csv_path.exists():
        try:
            old_chk = pd.read_csv(csv_path)
            for col in ["checked", "label_wrong", "cut_inside_word", "cut_bad_boundary", "corrected_label", "notes"]:
                if col in old_chk.columns:
                    old_map = dict(zip(old_chk["segment_id"].astype(str), old_chk[col]))
                    chk[col] = chk["segment_id"].astype(str).map(old_map).fillna(0 if "wrong" in col or "checked" in col or "inside" in col or "bad" in col else "")
        except Exception:
            pass

    for c, dv in [("label_wrong", 0), ("cut_inside_word", 0), ("cut_bad_boundary", 0), 
                 ("checked", 0), ("corrected_label", ""), ("notes", "")]:
        if c not in chk.columns:
            chk[c] = dv

    chk.to_csv(csv_path, index=False)
    return chk


def launch_manual_check(audio_samples: np.ndarray, video_dir: str | Path,
                        segments_df: Optional[pd.DataFrame] = None, sr: int = DEFAULT_SR):
    """Interactive checklist UI for full-video ground truth & human review verification."""
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    path = video_dir / "manual_check.csv"
    
    if segments_df is not None and (not path.exists() or len(pd.read_csv(path)) < len(segments_df)):
        sample_manual_check_window(segments_df, video_dir, minutes=None)
    elif not path.exists():
        raise FileNotFoundError(f"manual_check.csv not found in {video_dir} and segments_df was not provided.")
        
    chk = pd.read_csv(path)
    if "is_flagged" not in chk.columns:
        chk["is_flagged"] = (chk.get("needs_review", 0) == 1) | (chk.get("cut_tier", "") == "LOW")
    for col in ["label_wrong", "cut_inside_word", "cut_bad_boundary", "checked"]:
        if col not in chk.columns:
            chk[col] = 0
    for col in ["corrected_label", "notes"]:
        if col not in chk.columns:
            chk[col] = ""

    LABEL_COLORS = {
        "speech": "#2ca02c",
        "music": "#1f77b4",
        "mixed": "#ff7f0e",
        "silence": "#7f7f7f"
    }

    # 1. Scope selection: Flagged Only vs Full Video
    scope_radio = widgets.RadioButtons(
        options=[
            ("🔍 Flagged for Human Review Only (Labelling & Segmentation)", "flagged"),
            ("🎬 All Segments (Full Video)", "all")
        ],
        value="flagged" if int(chk["is_flagged"].sum()) > 0 else "all",
        description="Scope:",
        layout=widgets.Layout(width="100%")
    )

    stats_label = widgets.HTML()
    sel = widgets.Dropdown(description="Segment:", layout=widgets.Layout(width="80%"))
    
    # 2. Controls
    c_lbl = widgets.Checkbox(description="❌ Label is WRONG", value=False, indent=False, layout=widgets.Layout(width="220px"))
    corrected_drop = widgets.Dropdown(
        options=["speech", "music", "mixed", "silence"],
        description="Correct To:",
        layout=widgets.Layout(width="240px")
    )
    c_cut_word = widgets.Checkbox(description="✂️ Cut falls INSIDE a word (Word split risk)", value=False, indent=False, layout=widgets.Layout(width="310px"))
    c_cut_bad = widgets.Checkbox(description="⚠️ Unnatural / Bad Cut Boundary", value=False, indent=False, layout=widgets.Layout(width="280px"))
    txt_notes = widgets.Text(placeholder="Optional notes or reason for error...", description="Notes:", layout=widgets.Layout(width="90%"))

    b_save_next = widgets.Button(description="💾 Save & Next", button_style="success", icon="save", layout=widgets.Layout(width="150px"))
    b_mark_good = widgets.Button(description="✔ Mark Correct & Next", button_style="primary", icon="check", layout=widgets.Layout(width="180px"))
    b_prev = widgets.Button(description="⏮ Prev", layout=widgets.Layout(width="80px"))
    b_next = widgets.Button(description="Next ⏭", layout=widgets.Layout(width="80px"))

    info_out = widgets.Output()
    audio_out = widgets.Output()

    state = {
        "df": chk,
        "filtered_indices": []
    }

    def _sync_filter():
        df = state["df"]
        if scope_radio.value == "flagged":
            sub_idx = df[df["is_flagged"] == True].index.tolist()
            if not sub_idx:
                sub_idx = df.index.tolist()
        else:
            sub_idx = df.index.tolist()
        state["filtered_indices"] = sub_idx

        opts = []
        for i in sub_idx:
            r = df.loc[i]
            sid = r["segment_id"]
            lbl = str(r.get("label", r.get("primary_label", "speech"))).lower()
            tier = str(r.get("cut_tier", "HIGH")).upper()
            st, en, dur = float(r["start"]), float(r["end"]), float(r["duration"])
            is_fl = bool(r.get("is_flagged", False))
            is_done = bool(r.get("checked", 0) == 1)
            
            fl_tag = "⚠️" if is_fl else ""
            done_tag = "✔" if is_done else ""
            text = f"Seg {int(sid):04d} [{lbl.upper()}] [{tier}] {st:.1f}s–{en:.1f}s ({dur:.1f}s) {fl_tag} {done_tag}".strip()
            opts.append((text, i))
            
        sel.options = opts
        n_tot = len(df)
        n_fl = int(df["is_flagged"].sum())
        n_done = int((df["checked"] == 1).sum())
        stats_label.value = (
            f"<div style='font-size:13px; margin:4px 0; color:#555;'>"
            f"<b>Scope Segments:</b> {len(sub_idx)} &nbsp;|&nbsp; "
            f"<b>Flagged for Review (Labelling + Segmentation):</b> {n_fl}/{n_tot} ({100*n_fl/max(1,n_tot):.1f}%) &nbsp;|&nbsp; "
            f"<b>Checked:</b> {n_done}/{n_tot} ({100*n_done/max(1,n_tot):.1f}%)"
            f"</div>"
        )
        if sub_idx:
            if sel.value not in sub_idx:
                sel.value = sub_idx[0]

    def show(_=None):
        info_out.clear_output()
        audio_out.clear_output()
        if sel.value is None or sel.value not in state["df"].index:
            return
            
        i = sel.value
        r = state["df"].loc[i]
        
        c_lbl.value = bool(r.get("label_wrong", 0))
        c_cut_word.value = bool(r.get("cut_inside_word", 0))
        c_cut_bad.value = bool(r.get("cut_bad_boundary", 0))
        txt_notes.value = str(r.get("notes", "")) if pd.notnull(r.get("notes")) else ""
        
        lbl = str(r.get("label", r.get("primary_label", "speech"))).lower()
        if pd.notnull(r.get("corrected_label")) and str(r.get("corrected_label")).strip():
            corrected_drop.value = str(r.get("corrected_label")).strip()
        else:
            corrected_drop.value = lbl if lbl in ["speech", "music", "mixed", "silence"] else "speech"

        color = LABEL_COLORS.get(lbl, "#333")
        tier = str(r.get("cut_tier", "HIGH")).upper()
        conf = float(r.get("confidence", 1.0)) if pd.notnull(r.get("confidence")) else 1.0
        cert = float(r.get("cut_certainty", 1.0)) if pd.notnull(r.get("cut_certainty")) else 1.0
        is_fl = bool(r.get("is_flagged", False))
        prio = str(r.get("review_priority", "LOW")).upper()
        rev_reason = str(r.get("review_reason", "none"))
        cut_type = str(r.get("cut_type", "midpoint"))
        v_agree = str(r.get("voting_agreement", "3/3"))
        lead_ms = int(r.get("lead_ms", 150)) if pd.notnull(r.get("lead_ms")) else 150
        trail_ms = int(r.get("trail_ms", 150)) if pd.notnull(r.get("trail_ms")) else 150
        
        status_badge = f"<span style='background:{'#d9534f' if is_fl else '#5cb85c'}; color:white; padding:2px 8px; border-radius:4px; font-weight:bold;'>{'⚠️ REVIEW REQUIRED' if is_fl else '✅ AUTO-ACCEPTED'}</span>"
        checked_badge = "<span style='background:#0275d8; color:white; padding:2px 8px; border-radius:4px;'>✔ CHECKED</span>" if r.get("checked", 0) == 1 else ""

        html_info = f"""
        <div style="border:1px solid #ccc; border-radius:8px; padding:12px; margin:8px 0; background-color:#fcfcfc;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:16px; font-weight:bold;">Segment #{int(r['segment_id']):04d} &nbsp; ({float(r['start']):.2f}s – {float(r['end']):.2f}s, Dur: {float(r['duration']):.2f}s)</span>
                <div>{status_badge} &nbsp; {checked_badge}</div>
            </div>
            
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; font-size:13px; line-height:1.6;">
                <div style="background:#fff; border:1px solid #e0e0e0; border-radius:6px; padding:8px 12px;">
                    <div style="font-weight:bold; color:#333; margin-bottom:4px; border-bottom:1px solid #eee; padding-bottom:3px;">🏷️ SYSTEM LABELLING</div>
                    <div><b>Predicted Label:</b> <span style="background:{color}; color:white; padding:2px 7px; border-radius:3px; font-weight:bold;">{lbl.upper()}</span></div>
                    <div><b>Label Confidence:</b> {conf:.1%} &nbsp;|&nbsp; <b>Composition:</b> {r.get('label_composition', lbl)}</div>
                    <div><b>Source Regions:</b> {r.get('source_region_ids', 'N/A')}</div>
                </div>
                
                <div style="background:#fff; border:1px solid #e0e0e0; border-radius:6px; padding:8px 12px;">
                    <div style="font-weight:bold; color:#333; margin-bottom:4px; border-bottom:1px solid #eee; padding-bottom:3px;">✂️ SYSTEM SEGMENTATION</div>
                    <div><b>Certainty Tier:</b> <span style="font-weight:bold;">{tier}</span> (Score: {cert:.2f}) &nbsp;|&nbsp; <b>Priority:</b> {prio}</div>
                    <div><b>Cut Type:</b> {cut_type} &nbsp;|&nbsp; <b>Voting Agreement:</b> {v_agree}</div>
                    <div><b>Silence Margins:</b> Lead: {lead_ms}ms, Trail: {trail_ms}ms &nbsp;|&nbsp; <b>Reason:</b> <code style="color:#c7254e;">{rev_reason}</code></div>
                </div>
            </div>
        </div>
        """
        with info_out:
            display(HTML(html_info))
        with audio_out:
            st_samp = int(float(r["start"]) * sr)
            en_samp = int(float(r["end"]) * sr)
            display(Audio(audio_samples[st_samp:en_samp], rate=sr))

    def _save_feedback(mark_correct=False):
        i = sel.value
        if i is None or i not in state["df"].index:
            return
            
        if mark_correct:
            lbl_w = 0
            cut_w = 0
            cut_b = 0
            corr_lbl = ""
        else:
            lbl_w = int(c_lbl.value)
            cut_w = int(c_cut_word.value)
            cut_b = int(c_cut_bad.value)
            corr_lbl = corrected_drop.value if lbl_w else ""
            
        notes_val = txt_notes.value.strip()
        state["df"].loc[i, ["checked", "label_wrong", "cut_inside_word", "cut_bad_boundary", "corrected_label", "notes"]] = [
            1, lbl_w, cut_w, cut_b, corr_lbl, notes_val
        ]
        state["df"].to_csv(path, index=False)
        
        cur_list = state["filtered_indices"]
        if i in cur_list:
            pos = cur_list.index(i)
            if pos + 1 < len(cur_list):
                sel.value = cur_list[pos + 1]
            else:
                _sync_filter()
        else:
            _sync_filter()

    def on_prev(_):
        cur_list = state["filtered_indices"]
        if sel.value in cur_list:
            pos = cur_list.index(sel.value)
            if pos > 0:
                sel.value = cur_list[pos - 1]

    def on_next(_):
        cur_list = state["filtered_indices"]
        if sel.value in cur_list:
            pos = cur_list.index(sel.value)
            if pos + 1 < len(cur_list):
                sel.value = cur_list[pos + 1]

    scope_radio.observe(lambda _: _sync_filter(), names="value")
    sel.observe(show, names="value")
    b_save_next.on_click(lambda _: _save_feedback(mark_correct=False))
    b_mark_good.on_click(lambda _: _save_feedback(mark_correct=True))
    b_prev.on_click(on_prev)
    b_next.on_click(on_next)

    ui = widgets.VBox([
        widgets.HTML("<h3 style='margin-bottom:4px;'>🧪 Ground-Truth Manual Quality Check (Full Video & Review Queue)</h3>"),
        scope_radio,
        stats_label,
        widgets.HBox([b_prev, sel, b_next]),
        info_out,
        audio_out,
        widgets.HTML("<div style='font-weight:bold; margin-top:8px; margin-bottom:4px;'>Human Reviewer Annotations:</div>"),
        widgets.HBox([c_lbl, corrected_drop]),
        widgets.HBox([c_cut_word, c_cut_bad]),
        txt_notes,
        widgets.HBox([b_save_next, b_mark_good], layout=widgets.Layout(margin="8px 0"))
    ])
    
    _sync_filter()
    show()
    display(ui)
    return ui


def summarize_manual_check(video_dir: str | Path) -> pd.DataFrame:
    """Print counts for report + calibration of certainty tiers."""
    chk = pd.read_csv(Path(video_dir) / "manual_check.csv")
    done = chk[chk.checked == 1]
    print(f"Checked {len(done)}/{len(chk)} segments ({done.duration.sum() / 60:.1f} min)")
    if len(done):
        print(f"• Wrong region labels : {int(done.label_wrong.sum())} ({100 * done.label_wrong.mean():.1f}%)")
        print(f"• Cuts inside a word  : {int(done.cut_inside_word.sum())} ({100 * done.cut_inside_word.mean():.1f}%)")
        print("\nCut-tier calibration (bad cut share per tier):")
        display(done.groupby("cut_tier")["cut_inside_word"].agg(["count", "sum", "mean"]).rename(columns={"sum": "bad", "mean": "bad_rate"}))
    return done
