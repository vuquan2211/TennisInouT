# -*- coding: utf-8 -*-
"""
InouT – 4-Camera Live Viewer (no ffmpeg) + 10s Challenge on Cam A
"""

import os
import sys
import time
import cv2
import subprocess as sp
from collections import deque
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


# ===================== CONFIG =====================

ROOT = Path(r"C:\SAIT\TennisAI")
OUTPUT_DIR = ROOT / "input_video"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHALLENGE_CLIP = OUTPUT_DIR / "challenge_clip_10s.mp4"
REPLAY_SCRIPT = ROOT / "replay_10s.py"   # optional, can be missing

BUFFER_SEC = 10
TARGET_FPS = 30.0   # used for saving challenge clip

# ===================== CAPTURE WORKER =====================

class CaptureWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(int, object)  # cam_idx, frame (RGB)

    def __init__(self, cam_idx: int, url: str, parent=None):
        super().__init__(parent)
        self.cam_idx = cam_idx
        self.url = url
        self.cap = None
        self.running = False

        max_frames = int(BUFFER_SEC * TARGET_FPS)
        # For Cam A (index 0) we keep full buffer, others just small
        if cam_idx == 0:
            self.buffer = deque(maxlen=max_frames)
        else:
            self.buffer = deque(maxlen=int(TARGET_FPS * 2))  # 2s buffer just in case

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.url)

        if not self.cap.isOpened():
            print(f"[CaptureWorker] Cannot open camera {self.cam_idx}: {self.url}")
            return

        # Try to reduce latency a bit
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.msleep(10)
                continue

            if self.cam_idx == 0:
                # Keep full-resolution copy in buffer for challenge
                self.buffer.append(frame.copy())

            # Convert to RGB for display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(self.cam_idx, rgb)

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait(500)

    def get_buffer_copy(self):
        # Shallow copy for challenge saving
        return list(self.buffer)


# ===================== UI WIDGETS =====================

class VideoTile(QtWidgets.QLabel):
    def __init__(self, title="Cam"):
        super().__init__()
        self.setMinimumSize(360, 200)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText(f"{title}\nWaiting video…")
        self.setStyleSheet(
            "background:#101010; color:#e0e0e0; "
            "border:1px solid #333; border-radius:8px;"
        )

    def set_frame(self, frame):
        if frame is None:
            return
        h, w, ch = frame.shape
        qimg = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(
            pix.scaled(
                self.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )


class IpInputDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InouT - Camera IPs")
        self.setWindowIcon(QtGui.QIcon(r"C:\SAIT\Capstone\Logo\InouTLogo.png"))
        self.setMinimumWidth(520)

        form = QtWidgets.QFormLayout()
        placeholders = [
            "10.0.0.121:8080",
            "10.0.0.122:8080",
            "10.0.0.123:8080",
            "10.0.0.124:8080",
        ]
        self.edits = []
        for i in range(4):
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText(placeholders[i])
            form.addRow(f"Camera {i+1}", edit)
            self.edits.append(edit)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(btns)

    def _normalize(self, v: str):
        v = (v or "").strip()
        if not v:
            return None
        low = v.lower()
        if low.startswith(("http://", "https://", "rtsp://")):
            return v
        if v.isdigit():
            return f"http://127.0.0.1:{v}/video"
        if ":" in v:
            return f"http://{v}/video"
        return f"http://{v}:8080/video"

    def get_urls(self):
        urls = []
        for e in self.edits:
            raw = e.text().strip() or e.placeholderText()
            urls.append(self._normalize(raw))
        return urls


# ===================== CHALLENGE SAVER =====================

class ChallengeSaver(QtCore.QThread):
    finished_ok = QtCore.Signal(bool, str)  # ok, message

    def __init__(self, frames, parent=None):
        super().__init__(parent)
        self.frames = frames

    def run(self):
        if not self.frames:
            self.finished_ok.emit(False, "No frames buffered yet for Cam A.")
            return

        h, w = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(CHALLENGE_CLIP), fourcc, TARGET_FPS, (w, h))

        if not writer.isOpened():
            self.finished_ok.emit(False, "Cannot open VideoWriter for challenge clip.")
            return

        for f in self.frames:
            writer.write(f)
        writer.release()

        self.finished_ok.emit(True, f"Saved last 10s of Cam A to: {CHALLENGE_CLIP}")


# ===================== MAIN WINDOW =====================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, urls):
        super().__init__()
        self.setWindowTitle("TennisInouT - 4Cam Live (no ffmpeg) + 10s Challenge")
        self.setWindowIcon(QtGui.QIcon(r"C:\SAIT\Capstone\Logo\InouTLogo.png"))
        self.resize(1280, 780)

        cw = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(cw)
        self.setCentralWidget(cw)

        self.tiles = [VideoTile(f"Cam {chr(65+i)}") for i in range(4)]
        grid.addWidget(self.tiles[0], 0, 0)
        grid.addWidget(self.tiles[1], 0, 1)
        grid.addWidget(self.tiles[2], 1, 0)
        grid.addWidget(self.tiles[3], 1, 1)

        self.statusBar().showMessage("Ready")

        tb = self.addToolBar("Controls")

        self.actChallenge = QtGui.QAction("Challenge (save last 10s of Cam A)", self)
        self.actChallenge.triggered.connect(self.on_challenge)

        self.actQuit = QtGui.QAction("Quit", self)
        self.actQuit.triggered.connect(self.close)

        tb.addAction(self.actChallenge)
        tb.addSeparator()
        tb.addAction(self.actQuit)

        self.workers: list[CaptureWorker] = []

        for i in range(4):
            w = CaptureWorker(i, urls[i])
            w.frame_ready.connect(self.on_frame)
            w.start()
            self.workers.append(w)

        self.statusBar().showMessage("Live preview running (Cam A has 10s buffer for challenge).")

    @QtCore.Slot(int, object)
    def on_frame(self, cam_idx, frame):
        if 0 <= cam_idx < len(self.tiles):
            self.tiles[cam_idx].set_frame(frame)

    def on_challenge(self):
        cam_a = self.workers[0] if self.workers else None
        if not cam_a:
            self.statusBar().showMessage("Cam A not ready.")
            return

        frames = cam_a.get_buffer_copy()
        self.statusBar().showMessage("Saving last 10s of Cam A...")
        self.actChallenge.setEnabled(False)

        self.saver = ChallengeSaver(frames)
        self.saver.finished_ok.connect(self.on_challenge_done)
        self.saver.start()

    @QtCore.Slot(bool, str)
    def on_challenge_done(self, ok: bool, msg: str):
        self.actChallenge.setEnabled(True)
        self.statusBar().showMessage(msg, 5000)

        if ok and REPLAY_SCRIPT.is_file():
            try:
                cmd = [
                    sys.executable,
                    str(REPLAY_SCRIPT),
                    "--source",
                    str(CHALLENGE_CLIP),
                ]
                sp.Popen(cmd)
            except Exception as e:
                self.statusBar().showMessage(f"Replay script error: {e}", 5000)

    def closeEvent(self, e):
        for w in self.workers:
            w.stop()
        return super().closeEvent(e)


# ===================== ENTRY POINT =====================

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("InouT")
    app.setApplicationName("TennisInouT - 4Cam Live (no ffmpeg)")

    LOGO_PATH = r"C:\SAIT\Capstone\Logo\InouTLogo.png"
    if Path(LOGO_PATH).exists():
        pix_orig = QtGui.QPixmap(LOGO_PATH)
        if not pix_orig.isNull():
            pix = pix_orig.scaled(
                480, 480,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            splash = QtWidgets.QSplashScreen(pix)
            splash.show()
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(1200)
            splash.close()

    dlg = IpInputDialog()
    if dlg.exec() == QtWidgets.QDialog.Accepted:
        urls = dlg.get_urls()
        win = MainWindow(urls)
        win.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)
