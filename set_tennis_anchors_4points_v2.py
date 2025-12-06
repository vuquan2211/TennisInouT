# -*- coding: utf-8 -*-
"""
TennisAI – Set 4 anchor points for the mini-court.

Click order on the COURT IMAGE: 
    1. Bottom-left
    2. Bottom-right
    3. Top-right
    4. Top-left

After saving, this script creates:

- CALIB/tennis_anchors_4pts.json    (the 4 clicked points, in px + normalized)
- CALIB/anchors_tennis_19.json      (19 standard anchor points on a court canvas)
- CALIB/tennis_anchors_preview.jpg  (preview of the 4 clicked points on the image)
- CALIB/anchors19_preview.jpg       (REVIEW: grid + 19 anchors projected on the image)
"""

from pathlib import Path
import cv2, json, time, sys, numpy as np

# ======= PATHS =======
COURT_IMG   = r"C:\SAIT\TennisAI\Image\TennisCourtMap.jpeg"
OUT_DIR     = Path(r"C:\SAIT\TennisAI\CALIB")
JSON_4PTS   = OUT_DIR / "tennis_anchors_4pts.json"
PREV_4PTS   = OUT_DIR / "tennis_anchors_preview.jpg"
JSON_19PTS  = OUT_DIR / "anchors_tennis_19.json"
PREV_19PTS  = OUT_DIR / "anchors19_preview.jpg"

# ======= COURT SIZE (meters) =======
L_M, W_D_M, W_S_M, SVC_M = 23.77, 10.97, 8.23, 6.40  # length, doubles width, singles width, service-line dist.

# Canonical court canvas used to define court coordinates (pixels)
CW = 1000
CH = int(round(CW * (L_M / W_D_M)))   # keep correct L : W_D ratio

# ======= UI =======
N_POINTS    = 4
WIN_NAME    = "Click corners 1→4: BL, BR, TR, TL  |  Z=Undo  R=Reset  S=Save  Q=Quit"
LABEL_CLR   = (0, 200, 255)
WINDOW_SCALE = 0.60


def _resize_window(name, w, h):
    cv2.resizeWindow(name, int(w * WINDOW_SCALE), int(h * WINDOW_SCALE))


def build_anchors_19_bottom_up(cw, ch, Lm, Wd_m, Ws_m, svc_m):
    """
    Build 19 anchor points on the canonical court canvas, using a slight
    left/right perspective scaling so the overlay looks closer to broadcast angles.
    Returned points are in canvas pixel coordinates (cw x ch).
    """
    # Perspective factors (0.0–0.3 usually looks reasonable)
    left_scale  = 1.05   # expand left half
    right_scale = 0.95   # shrink right half

    inset_s = (Wd_m - Ws_m) / 2.0
    sx = (inset_s / Wd_m) * cw

    net_y   = ch / 2.0
    svc_px  = (svc_m / Lm) * ch
    y_svc_bot = net_y + svc_px
    y_svc_top = net_y - svc_px

    def L(x): return x * left_scale
    def R(x): return cw - (cw - x) * right_scale

    pts = []
    # 1..4 bottom baseline
    pts += [(L(0),         ch - 1),
            (L(sx),        ch - 1),
            (R(cw - sx),   ch - 1),
            (R(cw - 1),    ch - 1)]
    # 5..7 bottom service line
    pts += [(L(sx),        y_svc_bot),
            (cw / 2.0,     y_svc_bot),
            (R(cw - sx),   y_svc_bot)]
    # 8..12 net row
    pts += [(L(0),         net_y),
            (L(sx),        net_y),
            (cw / 2.0,     net_y),
            (R(cw - sx),   net_y),
            (R(cw - 1),    net_y)]
    # 13..15 top service line
    pts += [(L(sx),        y_svc_top),
            (cw / 2.0,     y_svc_top),
            (R(cw - sx),   y_svc_top)]
    # 16..19 top baseline
    pts += [(L(0),         0),
            (L(sx),        0),
            (R(cw - sx),   0),
            (R(cw - 1),    0)]

    return [(float(x), float(y)) for (x, y) in pts]


class ClickCollector:
    """
    Collect 4 corner clicks on COURT_IMG.
    Keyboard:
        Z / z  = undo last point
        R / r  = reset all points
        S / s  = save (only when 4 points)
        Q / q or ESC = cancel
    """
    def __init__(self, img):
        self.img0 = img.copy()
        self.img  = img.copy()
        self.pts  = []
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        _resize_window(WIN_NAME, img.shape[1], img.shape[0])
        cv2.setMouseCallback(WIN_NAME, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.pts) < N_POINTS:
            self.pts.append((x, y))
            self.redraw()

    def redraw(self):
        self.img = self.img0.copy()
        for i, (x, y) in enumerate(self.pts, 1):
            cv2.circle(self.img, (x, y), 6, LABEL_CLR, -1, cv2.LINE_AA)
            cv2.putText(self.img, str(i), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(WIN_NAME, self.img)

    def run(self):
        self.redraw()
        while True:
            k = cv2.waitKey(20) & 0xFF
            # Quit
            if k in (ord('q'), ord('Q'), 27):
                return None
            # Undo last
            if k in (ord('z'), ord('Z')) and self.pts:
                self.pts.pop()
                self.redraw()
            # Reset all
            if k in (ord('r'), ord('R')):
                self.pts.clear()
                self.redraw()
            # Save when we have all 4 points
            if k in (ord('s'), ord('S')) and len(self.pts) == N_POINTS:
                return self.pts


def save_json_4pts(img_path, img, pts):
    """
    Save 4 clicked corners into tennis_anchors_4pts.json
    and write a preview image with labels.
    """
    h, w = img.shape[:2]
    pts_norm = [(x / w, y / h) for x, y in pts]
    meta = {
        "created_at": int(time.time()),
        "image_path": img_path,
        "image_size": {"w": w, "h": h},
        "order_note": "Tennis court 4 corners: BL, BR, TR, TL",
        "points_px": pts,
        "points_norm": pts_norm,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_4PTS, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    prev = img.copy()
    for i, (x, y) in enumerate(pts, 1):
        cv2.circle(prev, (x, y), 6, LABEL_CLR, -1)
        cv2.putText(prev, str(i), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(PREV_4PTS), prev)
    print("✅ Saved anchors (4 pts):", JSON_4PTS)


def review_and_save_19pts(img, clicked_pts):
    """
    Build and save 19 canonical anchors, and render a REVIEW overlay.

    clicked_pts: [BL, BR, TR, TL] on COURT_IMG.
    Steps:
        - Build homography H: Court(canvas TL,TR,BR,BL) -> Image(BL,BR,TR,TL)
          (reordered as TL,TR,BR,BL).
        - Generate 19 anchors in canvas coordinates (CW x CH).
        - Save anchors_tennis_19.json (canvas coordinates only).
        - Warp an extended grid + project 19 anchors back onto COURT_IMG.
        - Save anchors19_preview.jpg.
    """
    H_img, W_img = img.shape[:2]

    # Court canvas corners in order TL, TR, BR, BL
    src_court = np.array(
        [[0, 0],
         [CW - 1, 0],
         [CW - 1, CH - 1],
         [0, CH - 1]],
        dtype=np.float32
    )

    # User clicked in order BL, BR, TR, TL -> reorder to TL, TR, BR, BL
    BL, BR, TR, TL = clicked_pts
    dst_img = np.array([TL, TR, BR, BL], dtype=np.float32)

    # Court(canvas) -> Image homography
    H_c2i, _ = cv2.findHomography(
        src_court,
        dst_img,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )
    if H_c2i is None:
        print("❌ Cannot compute homography Court→Image.")
        return

    # ===== Generate 19 anchors (bottom-up) in CANVAS coordinates =====
    anchors19_canvas = build_anchors_19_bottom_up(CW, CH, L_M, W_D_M, W_S_M, SVC_M)

    # Save 19 anchors JSON (defined in canvas coordinate system)
    meta19 = {
        "coord_mode": "pixel",
        "image_size": {"w": int(CW), "h": int(CH)},  # coordinate system size
        "court_size": {
            "L_m": float(L_M),
            "W_D_m": float(W_D_M),
            "SVC_m": float(SVC_M),
        },
        "anchors_19": [
            {"id": i + 1, "x": p[0], "y": p[1]}
            for i, p in enumerate(anchors19_canvas)
        ],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": (
            "19 anchors (bottom→top). Canvas CWxCH; aligned to court image "
            "via H_c2i for review overlay."
        ),
    }
    with open(JSON_19PTS, "w", encoding="utf-8") as f:
        json.dump(meta19, f, ensure_ascii=False, indent=2)
    print("✅ Saved anchors (19 pts meta):", JSON_19PTS)

    # ===== Build REVIEW overlay: extended grid + 19 anchors projected on image =====
    GRID_NX, GRID_NY, OUTER = 10, 18, 1
    cell_w, cell_h = CW / GRID_NX, CH / GRID_NY
    offx, offy = int(round(OUTER * cell_w)), int(round(OUTER * cell_h))
    CW_EXT, CH_EXT = int(CW + 2 * offx), int(CH + 2 * offy)

    grid = np.zeros((CH_EXT, CW_EXT, 3), np.uint8)
    step_x, step_y = max(1, int(round(cell_w))), max(1, int(round(cell_h)))

    # grid lines
    for x in range(0, CW_EXT + 1, step_x):
        cv2.line(grid, (x, 0), (x, CH_EXT - 1), (170, 170, 170), 1, cv2.LINE_AA)
    for y in range(0, CH_EXT + 1, step_y):
        cv2.line(grid, (0, y), (CW_EXT - 1, y), (170, 170, 170), 1, cv2.LINE_AA)

    # inner court rectangle
    cv2.rectangle(
        grid,
        (offx, offy),
        (offx + CW - 1, offy + CH - 1),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Warp extended grid to image (compensate for offset)
    T_off = np.array([[1, 0, -offx],
                      [0, 1, -offy],
                      [0, 0,  1   ]],
                     dtype=np.float64)
    H_ext_c2i = H_c2i @ T_off
    warped = cv2.warpPerspective(grid, H_ext_c2i, (W_img, H_img))
    vis = cv2.addWeighted(img, 0.72, warped, 0.28, 0)

    # Project 19 anchors to image for labeling
    pts = np.array(anchors19_canvas, dtype=np.float32).reshape(-1, 1, 2)
    pts_img = cv2.perspectiveTransform(pts, H_c2i).reshape(-1, 2)
    for i, (x, y) in enumerate(pts_img, start=1):
        xi, yi = int(round(x)), int(round(y))
        cv2.circle(vis, (xi, yi), 6, (80, 255, 80), -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            str(i),
            (xi + 6, yi - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            str(i),
            (xi + 6, yi - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (80, 255, 80),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(PREV_19PTS), vis)

    # Review window
    win = "REVIEW 19 points (press any key to close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    _resize_window(win, W_img, H_img)
    cv2.imshow(win, vis)
    cv2.waitKey(0)
    cv2.destroyWindow(win)


def main():
    img = cv2.imread(COURT_IMG)
    if img is None:
        sys.exit(f"❌ Cannot read court image: {COURT_IMG}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect 4 corners
    cc = ClickCollector(img)
    pts = cc.run()
    cv2.destroyAllWindows()

    if not pts:
        print("❌ Operation cancelled.")
        return

    # Save 4 points JSON + preview
    save_json_4pts(COURT_IMG, img, pts)

    # Compute review and save 19 anchors / preview
    review_and_save_19pts(img, pts)


if __name__ == "__main__":
    main()
