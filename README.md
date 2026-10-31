# Tamil AI Video Studio (Kaggle 2×T4, Open-Source Only)

Complete, audio-first, open-weights AI video studio engineered specifically for **Kaggle Dual NVIDIA Tesla T4 GPUs (`2 × 15 GB VRAM`, Turing `sm_75`)**. Takes a user prompt in **Tanglish**, **Tamil**, or **English** and produces a **30–60 s cinema-grade video** with a cloned **24 kHz Tamil voice (`AI4Bharat IndicF5`)**, **Whisper Auto-QA**, situation-matched instrumental music (`ACE-Step 1.5 Turbo`), environmental ambience (`MMAudio` / `LTX-2.3`), sidechain audio ducking (`ffmpeg`), and automatic background cloud sync to **Google Drive (`rclone` OAuth)**.

## 1. Setup Guide (Kaggle Secrets & Google Drive OAuth)
1. On your local machine (with a browser), run `rclone authorize "drive"` and copy the JSON OAuth token.
2. In your Kaggle Notebook (**Add-ons -> Secrets**), set:
   - `RCLONE_GDRIVE_TOKEN`: Paste the JSON string from `rclone authorize "drive"`.
   - `STUDIO_UI_PASSWORD`: Password/token required for the public `cloudflared` tunnel URL.
   - `HF_TOKEN`: Your Hugging Face read token (`hf_...`).
3. Enable **GPU T4 x 2** and **Internet ON**, then run all 7 ordered cells in `notebooks/tamil_ai_video_studio_kaggle.ipynb`.

## 2. Running Per-Stage Smoke Tests
```bash
python3 scripts/smoke_test_all.py --stage all
```
