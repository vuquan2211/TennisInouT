# -*- coding: utf-8 -*-
"""
challenge_call.py
----------------------------
- Play Video3_cut.mp4 at ~30fps.
- Always keep the last 10 seconds of frames in a buffer.
- Features:
    + Timeline (trackbar 'pos') to seek.
    + Play / Pause (Space).
    + Seek by dragging the timeline to any position.

- Press "C":
    1) Save the last 10 seconds as challenge_clip_10s.mp4
    2) Call replay_10s.py with --source = that 10s clip.

Press "Q" or ESC to quit.

Requirements:
  - replay_10s.py is in C:\SAIT\TennisAI
  - ROOT / input_video / CALIB / Image paths are same as v12 script
"""

import cv2
import sys
import subprocess
from collections import deque
from pathlib import Path

# ================== PATH CONFIG ==================
ROOT = Path(r"C:\SAIT\TennisAI")

VIDEO_DIR          = ROOT / "input_video"
DEFAULT_VIDEO      = VIDEO_DIR / "Video3_cut.mp4"

# 10s challenge clip will be saved here
CHALLENGE_CLIP_PATH = VIDEO_DIR / "challenge_clip_10s.mp4"

# v12 detect script
V12_SCRIPT = ROOT / "replay_10s.py"

# Calib + minimap (same as in v12)
CALIB_DIR    = ROOT / "CALIB"
MINIMAP_PATH = ROOT / "Image" / "TennisCourtMap.jpeg"

# Output video for the 10s clip
V12_OUT_PATH = ROOT / "outputs" / "Tennis_minimap_path_v12_challenge.mp4"

# ================== UI CONFIG ==================
WINDOW_NAME = "Tennis Video (C=Challenge, Space=Play/Pause, Q=Quit)"
SCALE       = 0.5  # 0.5 = half size, 1.0 = full size


# =============== HELPERS ===============

def save_clip_from_frames(frames, fps, out_path: Path) -> bool:
    """Save a list of BGR frames as an mp4 clip."""
    if not frames:
        print("[WARN] No frames to save for 10s clip.")
        return False

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        print(f"[ERROR] Cannot open VideoWriter for: {out_path}")
        return False

    for f in frames:
        writer.write(f)

    writer.release()
    print(f"[SAVE] Saved 10s clip to: {out_path}")
    return True


def call_detect_v12_on_clip(clip_path: Path):
    """Call replay_10s.py on the saved 10s clip."""
    if not clip_path.is_file():
        print(f"[WARN] Clip not found for detect: {clip_path}")
        return

    if not V12_SCRIPT.is_file():
        print(f"[WARN] v12 script not found: {V12_SCRIPT}")
        return

    print(f"[INFO] Running replay_10s on clip: {clip_path}")

    cmd = [
        sys.executable,
        str(V12_SCRIPT),
        "--source", str(clip_path),
        "--minimap", str(MINIMAP_PATH),
        "--calib", str(CALIB_DIR),
        "--out", str(V12_OUT_PATH),
        "--flip_y", "1",
        "--show"
    ]

    subprocess.run(cmd)
    print("[INFO] replay_10s finished on 10s clip.")


# =============== MAIN LOOP ===============

def main(video_path: Path = DEFAULT_VIDEO):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("[WARN] Cannot read FPS, fallback to 30fps.")
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("[WARN] Cannot read total frame count; timeline may be inaccurate.")
        total_frames = 1

    max_frames = int(fps * 10)  # last 10 seconds
    frame_buffer = deque(maxlen=max_frames)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Player state
    is_paused = False
    pending_seek = None
    updating_slider = False
    last_frame = None

    # ---------- Trackbar callback ----------
    def on_trackbar(pos):
        nonlocal pending_seek, is_paused, updating_slider
        if updating_slider:
            return
        pending_seek = int(pos)
        is_paused = True

    cv2.createTrackbar(
        "pos",
        WINDOW_NAME,
        0,
        max(total_frames - 1, 1),
        on_trackbar
    )

    print("[INFO] Start playing video.")
    print("[Keys]")
    print("  Space  - Play/Pause")
    print("  ← / →  - Step 1 frame (when paused)")
    print("  C      - Challenge: use last 10 seconds for replay_10s")
    print("  Q / ESC- Quit")
    print("")
    print("You can also close the window with the [X] button to quit.")

    while True:
        # ====== handle window close (click X) ======
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Window closed. Exiting.")
            break

        # ====== handle seek (timeline drag) ======
        if pending_seek is not None:
            target = max(0, min(pending_seek, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            frame_buffer.clear()

            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame after seek.")
                break

            frame_buffer.append(frame.copy())
            last_frame = frame.copy()

            updating_slider = True
            try:
                cv2.setTrackbarPos("pos", WINDOW_NAME, target)
            finally:
                updating_slider = False

            pending_seek = None

        else:
            # ====== normal play / pause read ======
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video.")
                    break
                frame_buffer.append(frame.copy())
                last_frame = frame.copy()
            else:
                if last_frame is None:
                    ret, frame = cap.read()
                    if not ret:
                        print("[INFO] End of video.")
                        break
                    frame_buffer.append(frame.copy())
                    last_frame = frame.copy()
                frame = last_frame

        # ====== display frame ====== 
        if SCALE != 1.0:
            display = cv2.resize(
                frame,
                (int(frame.shape[1] * SCALE), int(frame.shape[0] * SCALE)),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            display = frame

        cv2.imshow(WINDOW_NAME, display)

        # Update trackbar position when playing
        if not is_paused:
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_pos = max(0, min(current_pos, total_frames - 1))
            updating_slider = True
            try:
                cv2.setTrackbarPos("pos", WINDOW_NAME, current_pos)
            finally:
                updating_slider = False

        # ====== key handling ======
        key = cv2.waitKey(9) & 0xFF

        if key in (ord("q"), 27):  # Q or ESC
            print("[INFO] Quit by key.")
            break

        if key == ord(" "):  # Space: Play / Pause
            is_paused = not is_paused

        if key in (ord("c"), ord("C")):
            print("[INFO] Processing 10s challenge clip ...")
            frames_for_clip = list(frame_buffer)
            ok = save_clip_from_frames(frames_for_clip, fps, CHALLENGE_CLIP_PATH)
            if ok:
                call_detect_v12_on_clip(CHALLENGE_CLIP_PATH)

        if is_paused:
            # Right arrow: next frame
            if key == 83:
                curr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                next_f = min(curr + 1, total_frames - 1)
                pending_seek = next_f

            # Left arrow: previous frame
            if key == 81:
                curr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                prev_f = max(curr - 1, 0)
                pending_seek = prev_f

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
