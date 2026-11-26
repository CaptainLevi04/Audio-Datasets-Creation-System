# Audio Datasets Creation System (SautLab)

Local Arabic speech-dataset builder and desktop annotation GUI (`PySide6`). Downloads audio, classifies regions (`speech` / `mixed` / `music` / `silence`), segments at pauses with 2-detector voting (`Energy + Silero`), estimates SNR, runs sequential Dual ASR (`NVIDIA Arabic Conformer` + quantized `Audar-ASR-V1-Turbo` tuned for `>= 4 GB VRAM` GPUs), and exports reviewed transcripts/audio.

---

## 1. Requirements

- **Python 3.10–3.12** (tested on **Python 3.11**)
- **FFmpeg** installed and available on `PATH` (`ffmpeg -version`)
- **NVIDIA GPU (`>= 4 GB VRAM`, CUDA)** recommended for fast local ASR *(CPU and `--mock` mode work without a GPU)*

---

## 2. Install

```bash
git clone https://github.com/CaptainLevi04/Audio-Datasets-Creation-System.git
cd Audio-Datasets-Creation-System

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements-gui.txt
```

> **GPU note:** If `pip` installs CPU-only `torch`, install CUDA PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first, then re-run the `pip install` command above.

---

## 3. Run the Desktop GUI

```bash
python run_gui.py
```

### Quick Workflow in the GUI
1. Paste a YouTube link -> click **Download** *(or add a local audio/video file from the **Video** menu)*.
2. Click **Extract** (prepares 16 kHz mono WAV).
3. Click **Run Pipeline** (region classification -> 2-detector segmentation -> SNR -> Dual ASR).
   - *First run automatically downloads the local models and `llama.cpp` server into `~/.cache/`.*
4. Review/edit segments and transcripts, then click **Export CSV** *(or **Export** menu -> CSV / `.wav` + `.txt`)*.

### Try Without Downloads or GPU

```bash
python run_gui.py --mock
```

---

## 4. Useful Options (`run_gui.py`)

| Flag | Purpose |
|---|---|
| `--mock` | Launch with offline demo data (no downloads or GPU needed) |
| `--data-dir PATH` | Folder to store processed videos (default: `corpus_data`) |
| `--snr-threshold 20` | Clean/Noisy dB threshold (default: `25.0`; use `20` for YouTube/podcasts) |
| `--cookies cookies.txt` | Pass YouTube browser cookies if downloads are blocked |
| `--no-asr` | Run Steps 3–5 only (skip Dual ASR) |
| `--device cpu` / `cuda` | Force CPU or GPU for PyTorch/NeMo models |

---

## 5. Optional: CLI Pipeline Only (Steps 1–4)

```bash
python run_pipeline.py --demo
python run_pipeline.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --out-dir corpus_data
python run_pipeline.py --file "path/to/video.mp4" --out-dir corpus_data
```

---

## 6. Troubleshooting

- **Model download stalls on Windows (`Hugging Face`):**
  ```powershell
  $env:HF_HUB_DISABLE_XET="1"
  python run_gui.py
  ```
- **YouTube blocks download:** Export cookies from Chrome/Edge using a `cookies.txt` browser extension while signed in to YouTube, then run:
  ```bash
  python run_gui.py --cookies path/to/cookies.txt
  ```
