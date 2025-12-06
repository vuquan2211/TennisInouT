# -*- coding: utf-8 -*-
"""
Calibrate homography from 4 corners (frame -> court canvas).

Click order on the VIDEO FRAME: TL, TR, BR, BL  (4 outer doubles corners)

Outputs saved to CALIB/:
- H_frame_to_court_auto19.npy
- auto19_preview.png    (grid + outer cells)
- verify_19pts.png      (19 anchors overlay)
- H_meta.json           (for debugging the clicked corners)
"""

import os, json, time
import cv2
import numpy as np

# ================= PATHS =================
BASE_DIR  = r"C:\SAIT\TennisAI"
VIDEO     = os.path.join(BASE_DIR, "input_video", "Video3_cut.mp4")
CALIB_DIR = os.path.join(BASE_DIR, "CALIB")
os.makedirs(CALIB_DIR, exist_ok=True)

# ============== COURT CANVAS SIZE ==============
CW = 1000  # canvas width in pixels

# Use the same physical parameters as the anchors file:
# L_M: court length, W_D_M: doubles width, W_S_M: singles width, SVC_M: service-line distance from net
L_M, W_D_M, W_S_M, SVC_M = 23.77, 10.97, 8.23, 6.40
CH = int(round(CW * (L_M / W_D_M)))  # canvas height in pixels

# ========== PREVIEW GRID CONFIG ==========
GRID_NX     = 10
GRID_NY     = 18
OUTER_CELLS = 1  # number of extra grid cells around the court

# ========== UI ==========
UI_SCALE = 0.70
WIN  = "Click TL, TR, BR, BL  |  [A] Accept  [B] Reset  [Esc] Quit"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(img, text, pos, s=0.85, color=(255, 255, 0), thick=2):
    cv2.putText(img, text, pos, FONT, s, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, s, color, thick, cv2.LINE_AA)


def draw_points(img, pts, color=(0, 255, 255)):
    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), 8, color, -1, cv2.LINE_AA)


def pick_4_points(frame):
    """Let user click TL, TR, BR, BL on the frame. Returns list of 4 points or None if cancelled."""
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    Hf, Wf = frame.shape[:2]
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, int(Wf * UI_SCALE), int(Hf * UI_SCALE))
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        view = frame.copy()
        put_text(
            view,
            "Order: TL, TR, BR, BL  |  [B] Reset  [A] Accept  [Esc] Quit",
            (10, 30),
        )
        draw_points(view, pts)
        cv2.imshow(WIN, view)
        key = cv2.waitKey(10) & 0xFF

        # ESC or 'q' -> cancel
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            return None

        # 'B' -> reset points
        if key in (ord("b"), ord("B")):
            pts = []

        # 'A' -> accept if exactly 4 points
        if key in (ord("a"), ord("A")) and len(pts) == 4:
            cv2.destroyAllWindows()
            return pts


def build_anchors_19_court_px(cw, ch):
    """
    Build the 19 anchor points in court-space (pixels) using physical dimensions.
    Returns a list of (x, y) in the CW x CH canvas.
    """
    inset_s = (W_D_M - W_S_M) / 2.0  # side distance from doubles to singles line
    sx = (inset_s / W_D_M) * cw

    net_y = ch / 2.0
    svc_px = (SVC_M / L_M) * ch
    y_svc_top = net_y - svc_px
    y_svc_bot = net_y + svc_px

    pts = []
    # top baseline: 1,2,3,4
    pts += [(0, 0), (sx, 0), (cw - sx, 0), (cw - 1, 0)]
    # top service line: 5,6,7
    pts += [(sx, y_svc_top), (cw / 2.0, y_svc_top), (cw - sx, y_svc_top)]
    # net line: 8,9,10,11,12
    pts += [(0, net_y), (sx, net_y), (cw / 2.0, net_y), (cw - sx, net_y), (cw - 1, net_y)]
    # bottom service line: 13,14,15
    pts += [(sx, y_svc_bot), (cw / 2.0, y_svc_bot), (cw - sx, y_svc_bot)]
    # bottom baseline: 16,17,18,19
    pts += [(0, ch - 1), (sx, ch - 1), (cw - sx, ch - 1), (cw - 1, ch - 1)]

    return [(float(x), float(y)) for (x, y) in pts]


def main():
    cap = cv2.VideoCapture(VIDEO)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"❌ Cannot read: {VIDEO}")
    Hf, Wf = frame.shape[:2]

    pts4 = pick_4_points(frame)
    if pts4 is None:
        print("❌ Calibration cancelled.")
        return

    # src: frame TL,TR,BR,BL ; dst: canvas TL,TR,BR,BL
    src = np.array(pts4, dtype=np.float32)
    dst = np.array([[0, 0], [CW - 1, 0], [CW - 1, CH - 1], [0, CH - 1]], dtype=np.float32)

    H_f2c, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H_f2c is None:
        raise SystemExit("❌ Homography failed (H_f2c is None).")

    # ===== GRID overlay with extra OUTER cells =====
    cell_w = CW / GRID_NX
    cell_h = CH / GRID_NY
    offx = int(round(OUTER_CELLS * cell_w))
    offy = int(round(OUTER_CELLS * cell_h))
    CW_EXT = int(CW + 2 * offx)
    CH_EXT = int(CH + 2 * offy)

    grid_ext = np.zeros((CH_EXT, CW_EXT, 3), np.uint8)
    # vertical lines
    for i in range(0, CW_EXT + 1, max(1, int(round(cell_w)))):
        cv2.line(grid_ext, (i, 0), (i, CH_EXT - 1), (170, 170, 170), 1, cv2.LINE_AA)
    # horizontal lines
    for j in range(0, CH_EXT + 1, max(1, int(round(cell_h)))):
        cv2.line(grid_ext, (0, j), (CW_EXT - 1, j), (170, 170, 170), 1, cv2.LINE_AA)
    # inner court rectangle
    cv2.rectangle(
        grid_ext,
        (offx, offy),
        (offx + CW - 1, offy + CH - 1),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # warp grid to original frame
    H_c2f = np.linalg.inv(H_f2c)
    T_off = np.array([[1, 0, -offx], [0, 1, -offy], [0, 0, 1]], dtype=np.float64)
    H_ext_c2f = H_c2f @ T_off
    warped_ext = cv2.warpPerspective(grid_ext, H_ext_c2f, (Wf, Hf))
    overlay = cv2.addWeighted(frame, 0.72, warped_ext, 0.28, 0.0)

    # ===== VERIFY 19 ANCHORS =====
    anchors19 = build_anchors_19_court_px(CW, CH)
    court_pts = np.array(anchors19, dtype=np.float32).reshape(-1, 1, 2)
    pts19_img = cv2.perspectiveTransform(court_pts, H_c2f).reshape(-1, 2)

    verify = overlay.copy()
    for i, (x, y) in enumerate(pts19_img, start=1):
        xi, yi = int(round(x)), int(round(y))
        cv2.circle(verify, (xi, yi), 6, (80, 255, 80), -1, cv2.LINE_AA)
        cv2.putText(verify, str(i), (xi + 6, yi - 6), FONT, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(verify, str(i), (xi + 6, yi - 6), FONT, 0.8, (80, 255, 80), 2, cv2.LINE_AA)

    # ===== SAVE =====
    np.save(os.path.join(CALIB_DIR, "H_frame_to_court_auto19.npy"), H_f2c)
    cv2.imwrite(os.path.join(CALIB_DIR, "auto19_preview.png"), overlay)
    cv2.imwrite(os.path.join(CALIB_DIR, "verify_19pts.png"), verify)

    meta = {
        "clicked_order": "TL, TR, BR, BL (outer doubles corners)",
        "points_frame": pts4,
        "canvas_size": {"w": CW, "h": CH},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(CALIB_DIR, "H_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("✅ Saved homography, previews and H_meta.json to:", CALIB_DIR)

    # Show verification image
    win = "Verify 19 points  (press any key)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(Wf * UI_SCALE), int(Hf * UI_SCALE))
    cv2.imshow(win, verify)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
