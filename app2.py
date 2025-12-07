# -*- coding: utf-8 -*-
"""
InouT – 4-Camera Live Viewer (no ffmpeg)
---------------------------------------
• Splash screen with InouT logo
• Camera IP dialog (4 cams)
• 2x2 live viewer using OpenCV + PySide6
"""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
from PySide6 import QtCore, QtGui, QtWidgets


# ===================== Worker Thread =======================

class VideoWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(int, QtGui.QImage)
    error = QtCore.Signal(int, str)

    def __init__(self, cam_index: int, source: str, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self.source = source
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

    def run(self):
        self._running = True
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            self.error.emit(self.cam_index, f"[X] Cannot open video: {self.source}")
            return

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self.error.emit(self.cam_index, f"[X] Frame read failed on {self.source}")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            self.frame_ready.emit(self.cam_index, img)

    def close(self):
        self._running = False
        if self._cap:
            self._cap.release()
        super().quit()
        super().wait()


# ===================== Main Window ==========================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, urls: List[str], logo_path: Path):
        super().__init__()
        self.urls = urls
        self.logo_path = logo_path

        central = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout()
        central.setLayout(grid)
        self.setCentralWidget(central)

        self.labels: List[QtWidgets.QLabel] = []
        for i in range(4):
            lbl = QtWidgets.QLabel(f"Camera {i+1}")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet("background: #202020; border:1px solid #333;")
            lbl.setMinimumSize(480, 270)
            grid.addWidget(lbl, i // 2, i % 2)
            self.labels.append(lbl)

        self.workers: List[VideoWorker] = []
        for i, url in enumerate(self.urls):
            w = VideoWorker(i, url, self)
            w.frame_ready.connect(self.on_frame_ready)
            w.error.connect(self.on_worker_error)
            w.start()
            self.workers.append(w)

        if logo_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(logo_path)))

        self.setWindowTitle("InouT - 4Cam Live (no ffmpeg)")
        self.resize(1000, 600)

    def on_frame_ready(self, cam_index: int, img: QtGui.QImage):
        pix = QtGui.QPixmap.fromImage(img)
        self.labels[cam_index].setPixmap(pix.scaled(
            self.labels[cam_index].size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))

    def on_worker_error(self, cam_index: int, msg: str):
        self.labels[cam_index].setText(msg)

    def closeEvent(self, e):
        for w in self.workers:
            w.close()
        return super().closeEvent(e)


# === Small Dialog to enter Camera URLs ===
class InputDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InouT – Camera IPs")
        layout = QtWidgets.QFormLayout(self)

        self.edits: List[QtWidgets.QLineEdit] = []
        for i in range(4):
            edit = QtWidgets.QLineEdit(f"http://0.0.0.0:808{i+1}")
            self.edits.append(edit)
            layout.addRow(f"Camera {i+1}", edit)

        box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addRow(box)

    def get_urls(self) -> List[str]:
        return [e.text().strip() for e in self.edits]


# ======================== Entry Point ========================

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("InouT")
    app.setApplicationName("TennisInouT - 4Cam Live (no ffmpeg)")

    BASE_DIR = Path(__file__).resolve().parent
    LOGO_PATH = BASE_DIR / "Logo" / "InouTLogo.png"

    # Splash screen
    if LOGO_PATH.exists():
        pix = QtGui.QPixmap(str(LOGO_PATH))
        if not pix.isNull():
            pix = pix.scaled(
                480, 480,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            splash = QtWidgets.QSplashScreen(pix)
            splash.show()
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(1200)
            splash.close()

    dlg = InputDialog()
    if dlg.exec() != QtWidgets.QDialog.Accepted:
        sys.exit(0)

    urls = dlg.get_urls()

    win = MainWindow(urls, LOGO_PATH)
    win.show()

    sys.exit(app.exec())
