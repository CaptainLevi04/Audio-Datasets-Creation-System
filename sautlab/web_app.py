"""Gradio web annotation GUI for Kaggle (Step 8 Human Override).

Runs directly inside a Kaggle notebook session (inline + share URL) on top of
`sautlab.backend.corpus.CorpusBackend` and `sautlab.session.AnnotationSession`,
reading and writing the exact same `corpus_data/<video_id>/` files and audit
log (`changes.csv`) as the desktop PySide6 `SautLab` app.
"""
from __future__ import annotations

import inspect
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from . import export
from .backend.base import Backend
from .backend.corpus import CorpusBackend
from .models import LABELS, LABEL_TITLES, PIPELINE_STEPS, STATUS_TITLES, STEP_TITLES, fmt_time
from .session import AnnotationSession
from .textutil import compare_transcripts, highlight_html, normalize_ar

TABLE_COLS = ["#", "Start", "End", "Dur (s)", "Label", "SNR (dB)", "Quality", "Status", "Final Transcript"]


def _write_pcm16_wav(path: Path, samples: np.ndarray, sr: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples, dtype=np.float32)
    pcm = np.round(np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return str(path)


def _rtl_card(title: str, html_body: str, chosen: bool = False, color: str = "#3b82f6") -> str:
    border = f"2px solid {color}" if chosen else "1px solid #334155"
    badge = f" <span style='color:{color}; font-weight:700;'>[CHOSEN]</span>" if chosen else ""
    body = html_body if html_body else "<span style='color:#94a3b8'>No transcript</span>"
    return (
        f"<div style='border:{border}; border-radius:8px; padding:10px 12px; background:#0f172a; color:#e2e8f0;'>"
        f"<div style='font-size:12px; font-weight:700; margin-bottom:6px; color:#cbd5e1;'>{title}{badge}</div>"
        f"<div dir='rtl' align='right' style='font-size:16px; line-height:1.6;'>{body}</div>"
        f"</div>"
    )


def _steps_bar_html(session: AnnotationSession) -> str:
    v = session.video
    if v is None:
        return "<div>No video loaded.</div>"
    pills = []
    for k in PIPELINE_STEPS:
        st = (v.steps.get(k) or {}).get("state", "pending")
        bg = "#166534" if st == "done" else ("#92400e" if st == "running" else "#1e293b")
        fg = "#dcfce7" if st == "done" else ("#fef3c7" if st == "running" else "#94a3b8")
        mark = "✓ " if st == "done" else ("● " if st == "running" else "")
        pills.append(
            f"<span style='background:{bg}; color:{fg}; padding:3px 9px; border-radius:999px; "
            f"font-size:12px; margin-right:5px; display:inline-block;'>{mark}{STEP_TITLES[k]}</span>"
        )
    c = session.counts()
    save_txt = (
        "<span style='color:#fbbf24; font-weight:700;'>● Unsaved changes</span>"
        if session.dirty
        else f"<span style='color:#4ade80; font-weight:700;'>✓ Saved"
        + (f" ({session.last_saved})" if session.last_saved else "")
        + "</span>"
    )
    stats = (
        f"Accepted: <b>{c.get('accepted', 0)}</b> | "
        f"Needs Review: <b>{c.get('needs_review', 0)}</b> | "
        f"Difficult: <b>{c.get('difficult', 0)}</b> | "
        f"Total: <b>{len(session.segments)}</b>"
    )
    return (
        f"<div style='padding:8px 12px; border:1px solid #334155; border-radius:8px; background:#0b1220; color:#e2e8f0;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:6px;'>"
        f"<span><b>{v.title}</b> ({fmt_time(v.duration)}) &nbsp;|&nbsp; {stats}</span>"
        f"<span>{save_txt}</span>"
        f"</div>"
        f"<div>{''.join(pills)}</div>"
        f"</div>"
    )


def build_gradio_app(
    backend: Backend,
    user: str = "Team",
    temp_dir: str | Path | None = None,
    done_event: threading.Event | None = None,
):
    """Build the Gradio Blocks application for Step 8 review and annotation."""
    import gradio as gr

    session = AnnotationSession(backend, user=user)
    if session.videos:
        session.load_video(session.videos[0].id, auto_save=False)

    clip_dir = Path(temp_dir) if temp_dir else (Path("/tmp/sautlab_web_clips") if Path("/tmp").exists() else Path("._sautlab_web_clips"))
    clip_dir.mkdir(parents=True, exist_ok=True)
    counter = {"n": 0}
    lock = threading.RLock()

    def _cleanup_old_temp() -> None:
        files = sorted(clip_dir.glob("*"), key=lambda p: p.stat().st_mtime)
        for old in files[:-40]:
            try:
                old.unlink()
            except OSError:
                pass

    def _filtered_segments(status_filter: str, label_filter: str, quality_filter: str):
        out = []
        for s in session.segments:
            if status_filter == "conflicts" and s.status not in ("needs_review", "difficult"):
                continue
            elif status_filter in ("accepted", "needs_review", "difficult") and s.status != status_filter:
                continue
            if label_filter != "all" and s.label != label_filter:
                continue
            if quality_filter == "clean" and s.quality != "clean":
                continue
            if quality_filter == "noisy" and s.quality != "noisy":
                continue
            out.append(s)
        return out

    def _render(status_filter: str, label_filter: str, quality_filter: str, context_sec: float, notice: str = ""):
        counter["n"] += 1
        tag = counter["n"]
        _cleanup_old_temp()

        v_choices = [(f"{v.title} ({v.id})", v.id) for v in session.videos]
        v_val = session.video.id if session.video else None

        filtered = _filtered_segments(status_filter, label_filter, quality_filter)
        filtered_ids = [s.id for s in filtered]

        # Build table rows
        tbl_rows = []
        seg_choices = []
        for s in filtered:
            num = session.number_of(s.id)
            snr_str = f"{s.snr_mean:.1f}" if s.snr_mean is not None else "-"
            q_str = (s.quality or "-").capitalize() if s.quality else "-"
            st_str = STATUS_TITLES.get(s.status, s.status)
            preview = (s.final_text or s.asr_a or "")[:65]
            tbl_rows.append([
                num,
                f"{s.start:.2f}",
                f"{s.end:.2f}",
                f"{s.duration:.2f}",
                LABEL_TITLES.get(s.label, s.label),
                snr_str,
                q_str,
                st_str,
                preview,
            ])
            seg_choices.append((f"#{num:04d} [{s.label.upper()}] {s.start:.2f}-{s.end:.2f}s ({st_str})", s.id))

        # Ensure selected_id is valid
        if session.current is None and filtered:
            session.selected_id = filtered[0].id
        s = session.current
        dd_val = s.id if (s is not None and s.id in filtered_ids) else (filtered_ids[0] if filtered_ids else None)
        if dd_val is not None and (s is None or s.id != dd_val):
            s = session.select(dd_val)

        # Default inspector values
        plot_path = None
        wav_seg = wav_start_cut = wav_end_cut = None
        s_start = s_end = s_split = 0.0
        s_label = "speech"
        snr_html = "<div>No segment selected.</div>"
        card_a_html = _rtl_card(f"Model A ({session.asr_names[0]})", "")
        card_b_html = _rtl_card(f"Model B ({session.asr_names[1]})", "")
        cmp_html = ""
        final_txt = ""
        notes_txt = ""

        if s is not None and session.video is not None:
            plot_path = session.render_segment_plot(s.id, clip_dir / f"plot_{tag}.png", context_sec=max(1.5, context_sec + 1.0))
            seg_samples, sr = session.slice_audio(s.start, s.end)
            wav_seg = _write_pcm16_wav(clip_dir / f"seg_{tag}.wav", seg_samples, sr)
            st_samples, sr = session.slice_audio(s.start - context_sec, s.start + context_sec)
            wav_start_cut = _write_pcm16_wav(clip_dir / f"cut_s_{tag}.wav", st_samples, sr)
            en_samples, sr = session.slice_audio(s.end - context_sec, s.end + context_sec)
            wav_end_cut = _write_pcm16_wav(clip_dir / f"cut_e_{tag}.wav", en_samples, sr)

            s_start = round(s.start, 3)
            s_end = round(s.end, 3)
            s_split = round((s.start + s.end) / 2.0, 3)
            s_label = s.label

            e_str = f"{s.snr_energy:.1f} dB" if s.snr_energy is not None else "-"
            w_str = f"{s.snr_wada:.1f} dB" if s.snr_wada is not None else "-"
            m_str = f"{s.snr_mean:.1f} dB" if s.snr_mean is not None else "-"
            q_badge = (s.quality or "none").upper()
            st_badge = STATUS_TITLES.get(s.status, s.status)
            snr_html = (
                f"<div style='padding:8px 10px; border:1px solid #334155; border-radius:8px; background:#0f172a; color:#e2e8f0; font-size:13px;'>"
                f"<b>Segment #{session.number_of(s.id)}</b> &nbsp;|&nbsp; Status: <b>{st_badge}</b> &nbsp;|&nbsp; "
                f"Conf: <b>{s.confidence:.2f}</b><br/>"
                f"<b>Mean SNR:</b> {m_str} &nbsp;|&nbsp; <b>Energy:</b> {e_str} &nbsp;|&nbsp; <b>WADA:</b> {w_str} &nbsp;|&nbsp; "
                f"<b>Quality:</b> {q_badge} (auto: {s.quality_auto or '-'})"
                f"</div>"
            )

            if s.has_text:
                cmp = compare_transcripts(s.asr_a, s.asr_b) if (s.asr_a or s.asr_b) else None
                nf = normalize_ar(s.final_text)
                ch_a = bool(nf) and nf == normalize_ar(s.asr_a)
                ch_b = bool(nf) and nf == normalize_ar(s.asr_b)
                card_a_html = _rtl_card(
                    f"Model A ({session.asr_names[0]})",
                    highlight_html(s.asr_a, (cmp["tokens_a"], cmp["diff_a"]) if cmp else None, "#fbbf24"),
                    chosen=ch_a,
                    color="#60a5fa",
                )
                card_b_html = _rtl_card(
                    f"Model B ({session.asr_names[1]})",
                    highlight_html(s.asr_b, (cmp["tokens_b"], cmp["diff_b"]) if cmp else None, "#fbbf24"),
                    chosen=ch_b,
                    color="#c084fc",
                )
                if cmp is None:
                    cmp_html = "<div style='color:#94a3b8; font-size:13px;'>No ASR output yet for this segment.</div>"
                else:
                    stats = f"WER: {cmp['wer'] * 100:.1f}% | CER: {cmp['cer'] * 100:.1f}% | Diff words: {cmp['diff_words']}"
                    if cmp["agree"]:
                        cmp_html = f"<div style='color:#4ade80; font-weight:700;'>✓ Transcripts agree ({stats})</div>"
                    elif cmp["status"] == "needs_review":
                        cmp_html = f"<div style='color:#fbbf24; font-weight:700;'>⚠ Small difference ({stats})</div>"
                    else:
                        cmp_html = f"<div style='color:#f87171; font-weight:700;'>✖ Very different ({stats})</div>"
            else:
                card_a_html = _rtl_card(f"Model A ({session.asr_names[0]})", "<span style='color:#94a3b8'>No ASR for music/silence</span>")
                card_b_html = _rtl_card(f"Model B ({session.asr_names[1]})", "<span style='color:#94a3b8'>No ASR for music/silence</span>")
                cmp_html = "<div style='color:#94a3b8; font-size:13px;'>Music and silence segments bypass ASR.</div>"

            final_txt = s.final_text
            notes_txt = s.notes

        log_rows = [
            [e.get("time", ""), e.get("user", ""), e.get("segment") or "-", e.get("action", ""), e.get("detail", "")]
            for e in reversed(session.log[-100:])
        ]

        return (
            gr.update(choices=v_choices, value=v_val),
            _steps_bar_html(session),
            notice or "Ready.",
            filtered_ids,
            pd.DataFrame(tbl_rows, columns=TABLE_COLS),
            gr.update(choices=seg_choices, value=dd_val),
            plot_path,
            wav_seg,
            wav_start_cut,
            wav_end_cut,
            s_start,
            s_end,
            s_split,
            s_label,
            snr_html,
            card_a_html,
            card_b_html,
            cmp_html,
            final_txt,
            notes_txt,
            pd.DataFrame(log_rows, columns=["Time", "User", "Segment", "Action", "Detail"]),
        )

    with gr.Blocks(title="SautLab — Step 8 Speech Corpus Annotation GUI") as demo:
        filtered_ids_state = gr.State([])

        gr.Markdown("## SautLab — Step 8 Human Review & Override GUI (Integrated with `CorpusBackend`)")

        with gr.Row():
            video_dd = gr.Dropdown(label="Video", choices=[], interactive=True, scale=3)
            annotator_tb = gr.Textbox(label="Annotator", value=session.user, interactive=True, scale=1)
            btn_save = gr.Button("Save Changes", variant="primary", scale=1)
            btn_export = gr.Button("Export CSV + WAV/TXT", variant="secondary", scale=1)
            btn_finish = gr.Button("Finish Review & Continue", variant="stop", scale=1)

        steps_html = gr.HTML()
        status_msg = gr.Markdown("Ready.")

        with gr.Tabs():
            with gr.Tab("Annotate & Override"):
                with gr.Row():
                    status_f = gr.Dropdown(
                        label="Status Filter",
                        choices=[
                            ("All", "all"),
                            ("Conflicts Only (Needs Review + Difficult)", "conflicts"),
                            ("Needs Review", "needs_review"),
                            ("Difficult", "difficult"),
                            ("Accepted", "accepted"),
                        ],
                        value="all",
                        scale=2,
                    )
                    label_f = gr.Dropdown(
                        label="Label Filter",
                        choices=[("All", "all")] + [(LABEL_TITLES[l], l) for l in LABELS],
                        value="all",
                        scale=1,
                    )
                    quality_f = gr.Dropdown(
                        label="Quality Filter",
                        choices=[("All", "all"), ("Clean", "clean"), ("Noisy", "noisy")],
                        value="all",
                        scale=1,
                    )
                    ctx_slider = gr.Slider(label="Cut Context (±s)", minimum=0.5, maximum=4.0, step=0.5, value=1.5, scale=1)

                with gr.Row():
                    btn_prev = gr.Button("◀ Prev")
                    btn_next = gr.Button("Next ▶")
                    btn_prev_conf = gr.Button("⏮ Prev Conflict")
                    btn_next_conf = gr.Button("Next Conflict ⏭")
                    seg_dd = gr.Dropdown(label="Selected Segment", choices=[], interactive=True, scale=4)

                with gr.Row():
                    with gr.Column(scale=3):
                        seg_plot = gr.Image(label="Waveform & 0-8 kHz Spectrogram", type="filepath")
                        with gr.Row():
                            aud_seg = gr.Audio(label="Selected Segment Audio", type="filepath", interactive=False)
                            aud_cut_s = gr.Audio(label="Around Start Cut (±context)", type="filepath", interactive=False)
                            aud_cut_e = gr.Audio(label="Around End Cut (±context)", type="filepath", interactive=False)

                        with gr.Row():
                            in_start = gr.Number(label="Start (s)", precision=3, scale=1)
                            in_end = gr.Number(label="End (s)", precision=3, scale=1)
                            sync_nbr = gr.Checkbox(label="Sync neighbors", value=True, scale=1)
                            btn_move = gr.Button("Move Borders", scale=1)

                        with gr.Row():
                            in_split = gr.Number(label="Split at (s)", precision=3, scale=1)
                            btn_split = gr.Button("Split", scale=1)
                            btn_merge_prev = gr.Button("Merge w/ Prev", scale=1)
                            btn_merge_next = gr.Button("Merge w/ Next", scale=1)
                            btn_delete = gr.Button("Delete", variant="stop", scale=1)
                            btn_undo = gr.Button("Undo", scale=1)
                            btn_redo = gr.Button("Redo", scale=1)

                        with gr.Row():
                            add_start = gr.Number(label="Add Start (s)", value=0.0, precision=3, scale=1)
                            add_end = gr.Number(label="Add End (s)", value=5.0, precision=3, scale=1)
                            add_label = gr.Dropdown(label="Add Label", choices=list(LABELS), value="speech", scale=1)
                            btn_add = gr.Button("Add Segment", scale=1)

                    with gr.Column(scale=2):
                        snr_box = gr.HTML()
                        with gr.Row():
                            in_label = gr.Dropdown(label="Segment Label", choices=list(LABELS), value="speech", scale=2)
                            btn_label = gr.Button("Apply Label", scale=1)

                        with gr.Row():
                            btn_q_clean = gr.Button("Set Clean")
                            btn_q_noisy = gr.Button("Set Noisy")
                            btn_q_auto = gr.Button("Auto Quality")
                            btn_re_snr = gr.Button("Recompute SNR")

                        with gr.Row():
                            snr_thr_in = gr.Number(label="Global SNR Threshold (dB)", value=session.snr_threshold, precision=1, scale=2)
                            btn_snr_thr = gr.Button("Apply Threshold", scale=1)

                        card_a = gr.HTML()
                        card_b = gr.HTML()
                        cmp_box = gr.HTML()

                        with gr.Row():
                            btn_choose_a = gr.Button(f"Choose {session.asr_names[0]} (A)", variant="secondary")
                            btn_choose_b = gr.Button(f"Choose {session.asr_names[1]} (B)", variant="secondary")
                            btn_accept = gr.Button("Accept", variant="primary")
                            btn_diff = gr.Button("Mark Difficult", variant="stop")

                        tb_kwargs = {"label": "Final Transcript (editable RTL)", "lines": 3}
                        if "rtl" in inspect.signature(gr.Textbox.__init__).parameters:
                            tb_kwargs["rtl"] = True
                        in_final = gr.Textbox(**tb_kwargs)
                        btn_save_txt = gr.Button("Save Edited Final Transcript")
                        in_notes = gr.Textbox(label="Annotator Notes", lines=1)
                        btn_save_notes = gr.Button("Save Notes")

                seg_table = gr.Dataframe(
                    label="Segments Table (click any row to inspect)",
                    headers=TABLE_COLS,
                    interactive=False,
                    wrap=True,
                )

            with gr.Tab("Audit Log (changes.csv)"):
                log_table = gr.Dataframe(
                    label="Recorded Human Overrides",
                    headers=["Time", "User", "Segment", "Action", "Detail"],
                    interactive=False,
                    wrap=True,
                )

            with gr.Tab("Add Video / Run Steps 1-5"):
                gr.Markdown("Optionally add another YouTube URL or local audio/video path and run Steps 1-5 directly through `CorpusBackend`.")
                with gr.Row():
                    new_url = gr.Textbox(label="YouTube URL", scale=3)
                    btn_dl_url = gr.Button("Download & Run Steps 1-5", scale=1)
                with gr.Row():
                    new_local = gr.Textbox(label="Local / Kaggle Input Media Path", scale=3)
                    btn_add_local = gr.Button("Ingest & Run Steps 1-5", scale=1)

        all_outputs = [
            video_dd, steps_html, status_msg, filtered_ids_state, seg_table, seg_dd,
            seg_plot, aud_seg, aud_cut_s, aud_cut_e,
            in_start, in_end, in_split, in_label,
            snr_box, card_a, card_b, cmp_box, in_final, in_notes, log_table,
        ]
        f_inputs = [status_f, label_f, quality_f, ctx_slider]

        def _wrap(fn):
            def inner(*args):
                with lock:
                    try:
                        msg = fn(*args)
                    except Exception as e:
                        msg = f"**Error:** `{type(e).__name__}: {e}`"
                    return _render(args[-4], args[-3], args[-2], args[-1], notice=msg or "")
            return inner

        # Handlers (last 4 args are always status_f, label_f, quality_f, ctx_slider)
        def on_init(sf_v, lf_v, qf_v, cs_v):
            with lock:
                return _render(sf_v, lf_v, qf_v, cs_v, notice="Loaded project.")

        def on_user_change(u):
            session.user = (u or "Team").strip() or "Team"
            return f"Active annotator set to **{session.user}**."

        def on_video(vid, sf_v, lf_v, qf_v, cs_v):
            if vid and (session.video is None or session.video.id != vid):
                session.load_video(vid, auto_save=True)
                return f"Loaded video `{vid}`."
            return ""

        def on_filter(sf_v, lf_v, qf_v, cs_v):
            return "Filter updated."

        def on_select_dd(sid, sf_v, lf_v, qf_v, cs_v):
            if sid is not None:
                session.select(int(sid))
            return f"Selected segment #{session.number_of(session.selected_id)}."

        def on_table_select(f_ids, sf_v, lf_v, qf_v, cs_v, evt: gr.SelectData):
            with lock:
                row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
                if 0 <= row_idx < len(f_ids):
                    session.select(int(f_ids[row_idx]))
                return _render(sf_v, lf_v, qf_v, cs_v, notice=f"Selected segment #{session.number_of(session.selected_id)}.")

        def on_nav(step, conflict_only, sf_v, lf_v, qf_v, cs_v):
            if conflict_only:
                res = session.select_next_conflict(step)
                return "No remaining conflicts." if res is None else f"Jumped to conflict #{session.number_of(res.id)}."
            res = session.select_relative(step)
            return "" if res is None else f"Selected segment #{session.number_of(res.id)}."

        def on_move(st_v, en_v, sync_v, sf_v, lf_v, qf_v, cs_v):
            if session.current is None:
                return "No segment selected."
            session.set_times(session.current.id, float(st_v), float(en_v), sync_neighbors=bool(sync_v))
            return "Updated segment borders."

        def on_split(sp_v, sf_v, lf_v, qf_v, cs_v):
            if session.current is None:
                return "No segment selected."
            nid = session.split(session.current.id, float(sp_v))
            return f"Split segment into #{session.number_of(session.selected_id)} and #{session.number_of(nid)}."

        def on_merge(direction, sf_v, lf_v, qf_v, cs_v):
            if session.current is None:
                return "No segment selected."
            ok = session.merge(session.current.id, direction=direction)
            return "Merged segments." if ok else f"Cannot merge with {direction} neighbor."

        def on_delete(sf_v, lf_v, qf_v, cs_v):
            if session.current is None:
                return "No segment selected."
            session.delete(session.current.id)
            return "Deleted segment."

        def on_add(a_s, a_e, a_l, sf_v, lf_v, qf_v, cs_v):
            nid = session.add(float(a_s), float(a_e), str(a_l))
            return "Added new segment." if nid is not None else "Could not add segment (overlaps or too short)."

        def on_undo(sf_v, lf_v, qf_v, cs_v):
            return "Undid last edit." if session.undo() else "Nothing to undo."

        def on_redo(sf_v, lf_v, qf_v, cs_v):
            return "Redid edit." if session.redo() else "Nothing to redo."

        def on_label(lbl_v, sf_v, lf_v, qf_v, cs_v):
            if session.current is None:
                return "No segment selected."
            session.set_label(session.current.id, lbl_v)
            return f"Set label to `{lbl_v}`."

        def on_quality(q_v, sf_v, lf_v, qf_v, cs_v):
            if session.current is None:
                return "No segment selected."
            session.set_quality(session.current.id, q_v)
            return f"Updated Clean/Noisy quality to `{session.current.quality or 'auto'}`."

        def on_reprocess(sf_v, lf_v, qf_v, cs_v):
            if session.current is None:
                return "No segment selected."
            session.reprocess_segment(session.current.id)
            return "Recomputed Energy & WADA SNR for current cut."

        def on_snr_thr(thr_v, sf_v, lf_v, qf_v, cs_v):
            n = session.apply_snr_threshold(float(thr_v))
            return f"Applied {float(thr_v):g} dB threshold ({n} segment(s) updated)."

        def on_choose(eng, sf_v, lf_v, qf_v, cs_v):
            if session.current is None:
                return "No segment selected."
            session.choose_transcript(session.current.id, eng)
            return f"Accepted Model {'A' if eng == 'a' else 'B'} transcript."

        def on_status(st_v, sf_v, lf_v, qf_v, cs_v):
            if session.current is None:
                return "No segment selected."
            session.set_status(session.current.id, st_v)
            return f"Marked segment as `{st_v}`."

        def on_save_txt(txt_v, sf_v, lf_v, qf_v, cs_v):
            if session.current is None:
                return "No segment selected."
            session.set_final_text(session.current.id, txt_v or "")
            return "Saved edited final transcript."

        def on_save_notes(nt_v, sf_v, lf_v, qf_v, cs_v):
            if session.current is None:
                return "No segment selected."
            session.set_notes(session.current.id, nt_v or "")
            return "Saved notes."

        def on_save_project(u, sf_v, lf_v, qf_v, cs_v):
            session.user = (u or "Team").strip() or "Team"
            session.save()
            return f"Saved project to `{session.video.id}` (`segments.csv`, `segments_full.csv`, `segments/*.wav`, `transcripts/`, `changes.csv`)."

        def on_export_all(u, sf_v, lf_v, qf_v, cs_v):
            if session.video is None:
                return "No video loaded."
            session.user = (u or "Team").strip() or "Team"
            session.save()
            vdir = Path(session.backend.audio_path(session.video.id)).parent
            exp_dir = vdir / "exported_accepted"
            export.write_csv(vdir / f"{session.video.id}_review.csv", session.segments)
            n = export.export_segment_audio(session.backend.audio_path(session.video.id), exp_dir, session.video, session.segments)
            session.backend.mark_exported(session.video.id)
            session.refresh_videos()
            return f"Exported `{session.video.id}_review.csv` and {n} accepted `.wav + .txt` pairs to `{exp_dir}`."

        def on_finish(u, sf_v, lf_v, qf_v, cs_v):
            session.user = (u or "Team").strip() or "Team"
            if session.dirty and session.video is not None:
                session.save()
            if done_event is not None:
                done_event.set()
            return "Review saved and marked complete. Returning control to notebook for final packaging..."

        def on_ingest_url(url_v, sf_v, lf_v, qf_v, cs_v):
            if not (url_v or "").strip():
                return "Enter a YouTube URL."
            vi = session.backend.download(url_v.strip(), lambda f, m="": None)
            session.backend.extract_audio(vi.id, 16000, 1, lambda f, m="": None)
            session.backend.run_pipeline(vi.id, lambda f, m="": None)
            session.refresh_videos()
            session.load_video(vi.id, auto_save=True)
            return f"Ingested and processed `{vi.title}` (Steps 1-5 done; run Phase 3 in notebook for ASR)."

        def on_ingest_local(path_v, sf_v, lf_v, qf_v, cs_v):
            if not (path_v or "").strip():
                return "Enter a local or `/kaggle/input/...` media path."
            vi = session.backend.add_local_file(path_v.strip(), lambda f, m="": None)
            session.backend.extract_audio(vi.id, 16000, 1, lambda f, m="": None)
            session.backend.run_pipeline(vi.id, lambda f, m="": None)
            session.refresh_videos()
            session.load_video(vi.id, auto_save=True)
            return f"Ingested and processed `{vi.title}` (Steps 1-5 done; run Phase 3 in notebook for ASR)."

        demo.load(on_init, inputs=f_inputs, outputs=all_outputs)
        annotator_tb.change(on_user_change, inputs=[annotator_tb], outputs=[status_msg])
        video_dd.change(_wrap(on_video), inputs=[video_dd] + f_inputs, outputs=all_outputs)
        for comp in (status_f, label_f, quality_f, ctx_slider):
            comp.change(_wrap(on_filter), inputs=f_inputs, outputs=all_outputs)
        seg_dd.change(_wrap(on_select_dd), inputs=[seg_dd] + f_inputs, outputs=all_outputs)
        seg_table.select(on_table_select, inputs=[filtered_ids_state] + f_inputs, outputs=all_outputs)

        btn_prev.click(_wrap(lambda *a: on_nav(-1, False, *a)), inputs=f_inputs, outputs=all_outputs)
        btn_next.click(_wrap(lambda *a: on_nav(1, False, *a)), inputs=f_inputs, outputs=all_outputs)
        btn_prev_conf.click(_wrap(lambda *a: on_nav(-1, True, *a)), inputs=f_inputs, outputs=all_outputs)
        btn_next_conf.click(_wrap(lambda *a: on_nav(1, True, *a)), inputs=f_inputs, outputs=all_outputs)

        btn_move.click(_wrap(on_move), inputs=[in_start, in_end, sync_nbr] + f_inputs, outputs=all_outputs)
        btn_split.click(_wrap(on_split), inputs=[in_split] + f_inputs, outputs=all_outputs)
        btn_merge_prev.click(_wrap(lambda *a: on_merge("prev", *a)), inputs=f_inputs, outputs=all_outputs)
        btn_merge_next.click(_wrap(lambda *a: on_merge("next", *a)), inputs=f_inputs, outputs=all_outputs)
        btn_delete.click(_wrap(on_delete), inputs=f_inputs, outputs=all_outputs)
        btn_undo.click(_wrap(on_undo), inputs=f_inputs, outputs=all_outputs)
        btn_redo.click(_wrap(on_redo), inputs=f_inputs, outputs=all_outputs)
        btn_add.click(_wrap(on_add), inputs=[add_start, add_end, add_label] + f_inputs, outputs=all_outputs)

        btn_label.click(_wrap(on_label), inputs=[in_label] + f_inputs, outputs=all_outputs)
        btn_q_clean.click(_wrap(lambda *a: on_quality("clean", *a)), inputs=f_inputs, outputs=all_outputs)
        btn_q_noisy.click(_wrap(lambda *a: on_quality("noisy", *a)), inputs=f_inputs, outputs=all_outputs)
        btn_q_auto.click(_wrap(lambda *a: on_quality(None, *a)), inputs=f_inputs, outputs=all_outputs)
        btn_re_snr.click(_wrap(on_reprocess), inputs=f_inputs, outputs=all_outputs)
        btn_snr_thr.click(_wrap(on_snr_thr), inputs=[snr_thr_in] + f_inputs, outputs=all_outputs)

        btn_choose_a.click(_wrap(lambda *a: on_choose("a", *a)), inputs=f_inputs, outputs=all_outputs)
        btn_choose_b.click(_wrap(lambda *a: on_choose("b", *a)), inputs=f_inputs, outputs=all_outputs)
        btn_accept.click(_wrap(lambda *a: on_status("accepted", *a)), inputs=f_inputs, outputs=all_outputs)
        btn_diff.click(_wrap(lambda *a: on_status("difficult", *a)), inputs=f_inputs, outputs=all_outputs)
        btn_save_txt.click(_wrap(on_save_txt), inputs=[in_final] + f_inputs, outputs=all_outputs)
        btn_save_notes.click(_wrap(on_save_notes), inputs=[in_notes] + f_inputs, outputs=all_outputs)

        btn_save.click(_wrap(on_save_project), inputs=[annotator_tb] + f_inputs, outputs=all_outputs)
        btn_export.click(_wrap(on_export_all), inputs=[annotator_tb] + f_inputs, outputs=all_outputs)
        btn_finish.click(_wrap(on_finish), inputs=[annotator_tb] + f_inputs, outputs=all_outputs)

        btn_dl_url.click(_wrap(on_ingest_url), inputs=[new_url] + f_inputs, outputs=all_outputs)
        btn_add_local.click(_wrap(on_ingest_local), inputs=[new_local] + f_inputs, outputs=all_outputs)

    return demo, session


def launch_kaggle_gui(
    data_dir: str | Path = "corpus_data",
    user: str = "Team",
    snr_threshold: float = 20.0,
    snr_combine: str = "mean",
    cookies_file: str | None = None,
    share: bool = True,
    inline: bool = True,
    inline_height: int = 900,
    block_until_closed: bool = True,
):
    """Launch SautLab Step 8 web UI inside a Kaggle/Jupyter notebook session."""
    backend = CorpusBackend(
        data_dir=data_dir,
        classifier="auto",
        vad="auto",
        snr_threshold=snr_threshold,
        snr_combine=snr_combine,
        cookies_file=cookies_file,
    )
    done_event = threading.Event()
    demo, session = build_gradio_app(backend, user=user, done_event=done_event)

    print(f"Launching SautLab Web GUI for data_dir={Path(data_dir).resolve()} (annotator='{user}')...")
    launch_res = demo.launch(
        share=share,
        inline=inline,
        height=inline_height,
        prevent_thread_lock=True,
        quiet=True,
    )
    share_url = getattr(demo, "share_url", None)
    local_url = getattr(demo, "local_url", None)
    if isinstance(launch_res, tuple) and len(launch_res) >= 3:
        local_url = local_url or launch_res[1]
        share_url = share_url or launch_res[2]
    if share_url:
        print(f"Public Kaggle GUI URL: {share_url}")
    if local_url:
        print(f"Local URL: {local_url}")

    if block_until_closed:
        print("Click 'Finish Review & Continue' inside the GUI (or press Kaggle Stop) when done reviewing.")
        try:
            while not done_event.is_set():
                time.sleep(0.5)
            time.sleep(1.0)  # allow final HTTP response to reach the browser
        except KeyboardInterrupt:
            print("GUI wait interrupted by user; saving any unsaved changes...")
            if session.dirty and session.video is not None:
                session.save()
        finally:
            try:
                demo.close()
            except Exception:
                pass
    return demo, session
