# -*- coding: utf-8 -*-
"""
inout_decision.py

InouT Decision view:

- Input:
    * minimap (court image)
    * court_size (CW x CH in court-space)
    * H_c2mm (homography court -> minimap)
    * bounce_points_court (list of bounce points in court-space)
    * trail_court (~3 seconds of trajectory before the bounce)
    * fps (for reference only)

- Behavior:
    * Replays ball trajectory on the minimap:
        - Yellow path (ball path).
        - Green dots (ball positions).
        - Trim path to only the final segment around the bounce.
        - Upsample the trail for smoother ~3s animation.
        - Last replay frame: red dot at bounce position (same as replay_10s minimap).

    * After replay:
        - Soft zoom (~3s) from full minimap to the bounce red dot.
        - During zoom:
            + Path and green dots are NOT shown (clean court only).
            + A smaller red dot is drawn with fixed size (roughly actual ball size).

- IN / OUT decision:
    * Uses court-space (CW x CH) and anchors_tennis_19.json.
    * Classifies bounce as "IN" or "OUT".
    * Uses In.png / Out.png icons (no text).
    * Icon only appears after the zoom is complete (Hawk-Eye style).
"""

from pathlib import Path
import cv2
import numpy as np
import json

# ---- Defaults (used only when running this file directly for testing) ----
ROOT             = Path(r"C:\SAIT\TennisAI")
DEFAULT_MINIMAP  = str(ROOT / r"Image\TennisCourtMap.jpeg")
DEFAULT_CALIB    = str(ROOT / "CALIB")

DECISION_TARGET_H = 600   # target window height

# Colors (same as replay_10s)
PATH_COLOR        = (0, 255, 255)  # yellow (path)
DOT_COLOR         = (0, 255,   0)  # green (ball)
BOUNCE_COLOR      = (0,   0, 255)  # red (bounce)

TOTAL_REPLAY_MS   = 3000  # ~3 seconds for the full trail animation
ZOOM_TOTAL_MS     = 3000  # ~3 seconds zoom into the red dot
ZOOM_STEPS        = 60    # number of zoom steps (~60 frames ~ 3s, ~50 ms/frame)

# Red dot size:
BOUNCE_RADIUS_REPLAY = 4   # used during replay on full minimap
BOUNCE_RADIUS_ZOOM   = 12  # used in zoomed-in view (close to real ball size)

# ==== IN/OUT: uses anchors_tennis_19.json in CALIB ====
ANCHORS_19_JSON = ROOT / "CALIB" / "anchors_tennis_19.json"
_COURT_BOUNDS_CACHE = None  # (left_x, right_x, top_y, bottom_y) in court-space

# ==== IN / OUT ICONS ====
# Paths for icons in Capstone\Logo
IN_ICON_PATH    = Path(r"C:\SAIT\Capstone\Logo\In.png")
OUT_ICON_PATH   = Path(r"C:\SAIT\Capstone\Logo\Out.png")
_ICON_IN_CACHE  = None
_ICON_OUT_CACHE = None
ICON_MAX_HEIGHT = 140  # max icon height in the Decision window


# ---------------------------------------------------------
# Utility (standalone)
# ---------------------------------------------------------
def _read_json(p: Path):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_calib_for_decision(calib_dir: str, minimap_path: str):
    """
    Lightweight loader used only in _demo() when running this file directly.
    """
    cdir = Path(calib_dir)

    court_size = None
    j = cdir / "anchors_tennis_19.json"
    if j.exists():
        meta = _read_json(j)
        sz   = meta.get("image_size")
        if isinstance(sz, dict) and "w" in sz and "h" in sz:
            court_size = (int(sz["w"]), int(sz["h"]))

    if court_size is None:
        mm = cv2.imread(minimap_path)
        if mm is not None:
            court_size = (mm.shape[1], mm.shape[0])

    H_c2mm = None
    j4 = cdir / "tennis_anchors_4pts.json"
    if j4.exists():
        meta = _read_json(j4)
        pts4 = meta.get("points_px") or meta.get("image_points")
        if pts4 and len(pts4) == 4 and court_size is not None:
            BL, BR, TR, TL = pts4
            pts_mm = np.array([TL, TR, BR, BL], dtype=np.float32)
            orig_w = meta["image_size"]["w"]
            orig_h = meta["image_size"]["h"]
            sx = 1.0
            sy = 1.0
            if orig_w and orig_h:
                sx = float(court_size[0]) / float(orig_w)
                sy = float(court_size[1]) / float(orig_h)
            pts_mm[:, 0] *= sx
            pts_mm[:, 1] *= sy

            CW, CH = court_size
            src_court = np.array(
                [[0,      0],
                 [CW - 1, 0],
                 [CW - 1, CH - 1],
                 [0,      CH - 1]], dtype=np.float32
            )
            H_c2mm, _ = cv2.findHomography(src_court, pts_mm, cv2.RANSAC, 3.0)

    return court_size, H_c2mm


# ---------------------------------------------------------
# IN/OUT DECISION FROM COURT-SPACE (anchors_19)
# ---------------------------------------------------------
def _load_court_bounds_from_anchors(court_size):
    """
    Read CALIB/anchors_tennis_19.json and compute the outer court bounding box
    in court-space (CW x CH).

    Returns (left_x, right_x, top_y, bottom_y) in court-space.
    If file is missing / invalid -> fallback uses full court_size.
    """
    global _COURT_BOUNDS_CACHE

    if _COURT_BOUNDS_CACHE is not None:
        return _COURT_BOUNDS_CACHE

    cw, ch = court_size if court_size is not None else (None, None)
    left_x  = 0.0
    right_x = float(cw - 1) if cw is not None else 1.0
    top_y   = 0.0
    bot_y   = float(ch - 1) if ch is not None else 1.0

    try:
        meta = _read_json(ANCHORS_19_JSON)
        anchors = meta.get("anchors_19")
        sz_meta = meta.get("image_size")
        if anchors and sz_meta and "w" in sz_meta and "h" in sz_meta:
            cw_meta = float(sz_meta["w"])
            ch_meta = float(sz_meta["h"])

            # 1,4,16,19 (id-1) are four extreme points around the outer baseline
            xs = []
            ys = []
            idxs = [0, 3, 15, 18]  # anchors 1,4,16,19 (0-based)

            for i in idxs:
                if 0 <= i < len(anchors):
                    xs.append(float(anchors[i]["x"]))
                    ys.append(float(anchors[i]["y"]))

            if not xs:  # fallback: use all anchor points
                for a in anchors:
                    xs.append(float(a["x"]))
                    ys.append(float(a["y"]))

            left_x_meta  = min(xs)
            right_x_meta = max(xs)
            top_y_meta   = min(ys)
            bot_y_meta   = max(ys)

            if cw is not None and ch is not None and (cw != cw_meta or ch != ch_meta):
                sx = cw / cw_meta
                sy = ch / ch_meta
                left_x  = left_x_meta  * sx
                right_x = right_x_meta * sx
                top_y   = top_y_meta   * sy
                bot_y   = bot_y_meta   * sy
            else:
                left_x, right_x, top_y, bot_y = (
                    left_x_meta,
                    right_x_meta,
                    top_y_meta,
                    bot_y_meta,
                )
    except Exception:
        pass

    _COURT_BOUNDS_CACHE = (left_x, right_x, top_y, bot_y)
    return _COURT_BOUNDS_CACHE


def classify_inout_court(bounce_court_pt, court_size, margin_px=6):
    """
    Simple IN/OUT classifier in court-space:

    bounce_court_pt : (Xc, Yc) in court-space (CW x CH)
    court_size      : (CW, CH)
    margin_px       : pixel tolerance around outer lines

    Returns: "IN", "OUT", or None if unavailable.
    """
    if bounce_court_pt is None or court_size is None:
        return None

    Xc, Yc = bounce_court_pt
    left_x, right_x, top_y, bot_y = _load_court_bounds_from_anchors(court_size)

    if (left_x - margin_px <= Xc <= right_x + margin_px and
        top_y  - margin_px <= Yc <= bot_y  + margin_px):
        return "IN"
    else:
        return "OUT"


# ---------------------------------------------------------
# ICON IN/OUT
# ---------------------------------------------------------
def _get_decision_icon(decision, target_h):
    """
    Load + resize In/Out icon to fit into target_h (capped by ICON_MAX_HEIGHT).
    """
    global _ICON_IN_CACHE, _ICON_OUT_CACHE

    if decision not in ("IN", "OUT"):
        return None

    if decision == "IN":
        if _ICON_IN_CACHE is None:
            img = cv2.imread(str(IN_ICON_PATH), cv2.IMREAD_UNCHANGED)
            _ICON_IN_CACHE = img
        icon = _ICON_IN_CACHE
    else:
        if _ICON_OUT_CACHE is None:
            img = cv2.imread(str(OUT_ICON_PATH), cv2.IMREAD_UNCHANGED)
            _ICON_OUT_CACHE = img
        icon = _ICON_OUT_CACHE

    if icon is None:
        return None

    h, w = icon.shape[:2]
    max_h = min(ICON_MAX_HEIGHT, int(target_h * 0.25))
    if h > max_h:
        scale = max_h / float(h)
        icon = cv2.resize(icon, (int(w * scale), max_h), interpolation=cv2.INTER_AREA)

    return icon


def _overlay_icon_rgba(bg, icon_rgba, x, y):
    """
    Overlay RGBA icon onto bg at (x, y).
    """
    ih, iw = icon_rgba.shape[:2]
    h, w = bg.shape[:2]

    if x >= w or y >= h:
        return bg

    x2 = min(x + iw, w)
    y2 = min(y + ih, h)
    roi_w = x2 - x
    roi_h = y2 - y
    if roi_w <= 0 or roi_h <= 0:
        return bg

    icon = icon_rgba[0:roi_h, 0:roi_w, :]
    if icon.shape[2] == 4:
        alpha = icon[:, :, 3] / 255.0
        alpha = alpha[..., None]
        fg = icon[:, :, :3].astype(np.float32)
        bg_roi = bg[y:y+roi_h, x:x+roi_w, :].astype(np.float32)
        out = alpha * fg + (1 - alpha) * bg_roi
        bg[y:y+roi_h, x:x+roi_w, :] = out.astype(np.uint8)
    else:
        bg[y:y+roi_h, x:x+roi_w, :] = icon[:, :, :3]
    return bg


def draw_decision_icon(img, decision, pos=(40, 40)):
    """
    Draw IN or OUT icon on img at pos.
    If icon cannot be loaded or decision is None -> returns original image.
    """
    if decision not in ("IN", "OUT"):
        return img

    h = img.shape[0]
    icon = _get_decision_icon(decision, h)
    if icon is None:
        return img

    x, y = pos
    return _overlay_icon_rgba(img, icon, x, y)


# ---------------------------------------------------------
# MAIN FUNCTION: called from replay_10s
# ---------------------------------------------------------
def inout_decision(minimap,
                   court_size,
                   H_c2mm,
                   bounce_points_court,
                   trail_court=None,
                   fps=30.0,
                   win_name="InouT Decision"):
    """
    minimap : np.ndarray (BGR)
    court_size : (Wc, Hc) or None
    H_c2mm : homography court->minimap
    bounce_points_court : list[(Xc, Yc)] – typically just one point
    trail_court : list[(Xc, Yc) or None] – ~3s trajectory before bounce
    fps : for reference only (TOTAL_REPLAY_MS is fixed)
    """

    if minimap is None:
        print("⚠️ [InouT Decision] minimap is None, nothing to show.")
        return

    base = minimap.copy()
    mh, mw = base.shape[:2]

    # ====== IN/OUT decision in court-space ======
    bounce_court_last = bounce_points_court[-1] if bounce_points_court else None
    decision = classify_inout_court(bounce_court_last, court_size, margin_px=6)

    # 1) Convert bounce point(s) -> minimap
    pts_bounce_mm = []
    if bounce_points_court:
        for pt in bounce_points_court:
            if pt is None:
                continue
            Xc, Yc = pt
            if H_c2mm is not None:
                arr = np.array([[[Xc, Yc]]], dtype=np.float32)
                try:
                    mx, my = cv2.perspectiveTransform(arr, H_c2mm)[0, 0]
                    if np.isfinite(mx) and np.isfinite(my):
                        ix = int(np.clip(round(mx), 0, mw - 1))
                        iy = int(np.clip(round(my), 0, mh - 1))
                        pts_bounce_mm.append((ix, iy))
                except Exception:
                    continue
            else:
                if court_size is not None:
                    CW, CH = court_size
                else:
                    CW, CH = mw, mh
                ix = int(np.clip(round(Xc / max(CW, 1) * mw), 0, mw - 1))
                iy = int(np.clip(round(Yc / max(CH, 1) * mh), 0, mh - 1))
                pts_bounce_mm.append((ix, iy))

    # 2) Convert trail_court -> minimap trail
    mm_trail = None
    if trail_court and len(trail_court) > 1:
        mm_trail = []
        for pt in trail_court:
            if pt is None:
                mm_trail.append(None)
                continue
            Xc, Yc = pt
            if H_c2mm is not None:
                arr = np.array([[[Xc, Yc]]], dtype=np.float32)
                try:
                    mx, my = cv2.perspectiveTransform(arr, H_c2mm)[0, 0]
                    if np.isfinite(mx) and np.isfinite(my):
                        ix = int(np.clip(round(mx), 0, mw - 1))
                        iy = int(np.clip(round(my), 0, mh - 1))
                        mm_trail.append((ix, iy))
                    else:
                        mm_trail.append(None)
                except Exception:
                    mm_trail.append(None)
            else:
                if court_size is not None:
                    CW, CH = court_size
                else:
                    CW, CH = mw, mh
                ix = int(np.clip(round(Xc / max(CW, 1) * mw), 0, mw - 1))
                iy = int(np.clip(round(Yc / max(CH, 1) * mh), 0, mh - 1))
                mm_trail.append((ix, iy))

    # ==== Trim trail: keep only last segment around bounce ====
    if mm_trail:
        n = len(mm_trail)

        if pts_bounce_mm:
            bx, by = pts_bounce_mm[-1]
            best_i = None
            best_d = None
            for i, p in enumerate(mm_trail):
                if p is None:
                    continue
                dx = p[0] - bx
                dy = p[1] - by
                d2 = dx * dx + dy * dy
                if best_d is None or d2 < best_d:
                    best_d = d2
                    best_i = i

            if best_i is not None:
                desired = max(8, int(n * 0.25))
                start = max(0, best_i - desired + 1)
                mm_trail = mm_trail[start:best_i+1]
        else:
            cut_len = max(8, int(n * 0.25))
            mm_trail = mm_trail[-cut_len:]

    # ==== Upsample trail for smoother animation ====
    if mm_trail:
        UPSAMPLE = 3
        dense = []
        prev = None
        for p in mm_trail:
            if p is None:
                if prev is not None:
                    dense.append(prev)
                dense.append(None)
                prev = None
                continue
            if prev is None:
                dense.append(p)
            else:
                for s in range(1, UPSAMPLE + 1):
                    t = s / float(UPSAMPLE + 1)
                    ix = int(round(prev[0] + t * (p[0] - prev[0])))
                    iy = int(round(prev[1] + t * (p[1] - prev[1])))
                    dense.append((ix, iy))
                dense.append(p)
            prev = p
        mm_trail = dense

    # 3) If no trail -> static minimap + red dot + icon (no zoom)
    if not mm_trail:
        canvas = base.copy()
        for (ix, iy) in pts_bounce_mm:
            cv2.circle(canvas,
                       (ix, iy),
                       BOUNCE_RADIUS_REPLAY,
                       BOUNCE_COLOR,
                       -1,
                       cv2.LINE_AA)

        canvas = draw_decision_icon(canvas, decision)

        scale = DECISION_TARGET_H / float(mh) if mh > 0 else 1.0
        if scale <= 0:
            scale = 1.0
        view = cv2.resize(
            canvas,
            (int(mw * scale), int(mh * scale)),
            interpolation=cv2.INTER_NEAREST
        )
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, view)
        print("[InouT Decision] No trail_court: static minimap + red dot. Press ESC/Q to close.")
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyWindow(win_name)
        return

    # 4) Replay animation (~3 seconds, no icon yet)
    n = len(mm_trail)
    if n <= 1:
        return inout_decision(minimap, court_size, H_c2mm, bounce_points_court,
                              trail_court=None, fps=fps, win_name=win_name)

    delay_ms = max(10, int(TOTAL_REPLAY_MS / float(n)))

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    last_frame = None
    bounce_mm  = pts_bounce_mm[-1] if pts_bounce_mm else None

    for i in range(n):
        canvas = base.copy()

        # Yellow path
        prev = None
        for j in range(i + 1):
            pj = mm_trail[j]
            if pj is None:
                prev = None
                continue
            if prev is not None:
                cv2.line(canvas,
                         prev,
                         pj,
                         PATH_COLOR,
                         2,
                         cv2.LINE_AA)
            prev = pj

        # Ball (green dot)
        cur = mm_trail[i]
        if cur is not None:
            cv2.circle(canvas,
                       cur,
                       5,
                       DOT_COLOR,
                       -1,
                       cv2.LINE_AA)

        # Last frame of replay: red dot at bounce location (full minimap)
        if (bounce_mm is not None) and (i == n - 1):
            cv2.circle(canvas,
                       bounce_mm,
                       BOUNCE_RADIUS_REPLAY,
                       BOUNCE_COLOR,
                       -1,
                       cv2.LINE_AA)

        # No IN/OUT icon during replay phase

        scale_full = DECISION_TARGET_H / float(mh) if mh > 0 else 1.0
        if scale_full <= 0:
            scale_full = 1.0
        view = cv2.resize(
            canvas,
            (int(mw * scale_full), int(mh * scale_full)),
            interpolation=cv2.INTER_NEAREST
        )

        last_frame = view.copy()

        cv2.imshow(win_name, view)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            cv2.destroyWindow(win_name)
            return
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(win_name)
            return

    # ==== Soft zoom (~3 seconds) after red dot appears ====
    # During zoom: clean court (no path / green dots), only red dot + final icon at the end.
    zoom_last_frame = None
    if bounce_mm is not None:
        zoom_base = base.copy()  # empty court
        cx, cy = bounce_mm
        h_full, w_full = zoom_base.shape[:2]

        # Start from full minimap
        half_start = max(cx, cy, w_full - cx, h_full - cy)
        # End with a window ~15% of the shorter edge
        half_final = int(0.15 * min(w_full, h_full))

        zoom_delay = max(15, int(ZOOM_TOTAL_MS / float(ZOOM_STEPS)))

        for k in range(ZOOM_STEPS + 1):
            t = k / float(ZOOM_STEPS)
            half = int((1.0 - t) * half_start + t * half_final)

            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w_full, cx + half)
            y2 = min(h_full, cy + half)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = zoom_base[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            scale_h = DECISION_TARGET_H / float(crop.shape[0])
            view_zoom = cv2.resize(
                crop,
                (int(crop.shape[1] * scale_h), DECISION_TARGET_H),
                interpolation=cv2.INTER_NEAREST
            )

            # New red dot with fixed radius on zoomed view
            bx_crop = cx - x1
            by_crop = cy - y1
            bx_view = int(round(bx_crop * scale_h))
            by_view = int(round(by_crop * scale_h))
            if 0 <= bx_view < view_zoom.shape[1] and 0 <= by_view < view_zoom.shape[0]:
                cv2.circle(view_zoom,
                           (bx_view, by_view),
                           BOUNCE_RADIUS_ZOOM,
                           BOUNCE_COLOR,
                           -1,
                           cv2.LINE_AA)

            # No IN/OUT icon yet during zoom
            zoom_last_frame = view_zoom.copy()
            cv2.imshow(win_name, view_zoom)

            key = cv2.waitKey(zoom_delay) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                cv2.destroyWindow(win_name)
                return
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyWindow(win_name)
                return

    # After zoom: overlay IN/OUT icon on final frame
    if zoom_last_frame is not None:
        final_frame = zoom_last_frame.copy()
    else:
        final_frame = last_frame.copy() if last_frame is not None else base.copy()

    final_frame = draw_decision_icon(final_frame, decision)

    # 5) Hold final frame until user closes
    print("[InouT Decision] Replay + zoom finished. Showing IN/OUT. Press ESC/Q to close.")
    while True:
        cv2.imshow(win_name, final_frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(win_name)


# ---------------------------------------------------------
# Small demo when running this file directly
# ---------------------------------------------------------
def _demo():
    minimap = cv2.imread(DEFAULT_MINIMAP)
    if minimap is None:
        print(f"❌ Cannot load minimap from {DEFAULT_MINIMAP}")
        return

    court_size, H_c2mm = _load_calib_for_decision(DEFAULT_CALIB,
                                                  DEFAULT_MINIMAP)
    if court_size is None:
        court_size = (minimap.shape[1], minimap.shape[0])

    # Simple diagonal trail demo, then use the last point as a fake bounce
    trail = []
    for t in range(40):
        trail.append((court_size[0] * (0.1 + 0.8 * t / 39.0),
                      court_size[1] * (0.2 + 0.5 * t / 39.0)))
    bounce = [trail[-1]]

    inout_decision(minimap, court_size, H_c2mm, bounce, trail_court=trail, fps=30.0)

if __name__ == "__main__":
    _demo()
