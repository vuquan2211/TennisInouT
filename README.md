# TennisAI – Challenge Replay System

A lightweight tennis challenge system that records a 10-second clip from a live video source and automatically launches a slow-motion replay with minimap and bounce detection. Built with Python, OpenCV, and YOLOv8.

---

## Features
- Live video playback
- Save last 10 seconds (hotkey: C)
- Auto-run replay detection (replay_10s.py)
- Minimap and bounce visualization
- Ball trajectory detection
- Timeline scrub control (seek bar)
- Keyboard controls (play, pause, step)

---

# Installation Guide (New Machine)

## 1. Clone the repository
```bash
git clone https://github.com/vuquan2211/TennisAI.git
cd TennisAI

## 2. Install Python
Install Python 3.10–3.12

## 3. Create virtual environment
python -m venv venv

Activate environment
venv\Scripts\activate

## 4. Install dependencies
pip install -r requirements.txt

## 5. Download required assets (Google Drive)
Download all project assets from Google Drive:
https://drive.google.com/drive/folders/1cFSmZufbrbPHkSiV3jY7Z6fPoJQZzxdZ?usp=drive_link
This folder contains:
CALIB
Image
input_video
Logo
runs (YOLO models)
outputs
After downloading and extracting, copy all of these folders directly into:
C:\SAIT\TennisAI

## 6. Clone the repository onto a new machine
cd C:\SAIT
git clone https://github.com/vuquan2211/TennisAI.git
cd TennisAI

## 7. Run the application
cd C:\SAIT\TennisAI
python challenge_call.py
