# -*- coding: utf-8 -*-
"""
replay_10s.py
YOLOv8 + Snap-ROI + Minimap (Homography) + Bounce detection (parabola-refined)

v12 = a balance between v10 (strict) and v11 (loose):

- Candidate bounce:
    + Peak detector on y-axis (frame-space)
    + Fallback using vy
    + Kink detector in court-space (trajectory change on minimap)
    + Merge nearby candidates (merge_close_indices)

- Bounce decision:
    1) Filter out static false bounces (YOLO false positives).
    2) Use v10 kinematics classifier (ratio + angle + vy sign change). If "bounce" → accept.
    3) Fallback: if classifier rejects but parabola looks good & not “hit-like”
       (parabounce_ok + parabounce_hitlike) → still accept as bounce.

- Refine bounce timestamp using parabola (sub-frame).
- Record bounce (IN/OUT) on big frame & minimap (persistent).

- Detection runs headless, then:
    * Open an interactive replay player with:
      - Yellow timeline + red dot bounces.
      - Seekable "Timeline" trackbar.
      - InouT logo at bottom-right (overlay only, not written into mp4).
    * CLICK InouT logo:
        - Find bounce closest to current frame (on timeline).
        - Call inout_decision to open a minimap window showing exactly that one red dot.
"""

from pathlib import Path
import argparse, time, json, sys, re
import cv2
import numpy as np
from ultralytics import YOLO

import inout_decision  # inout_decision.py in the same folder

# ===================== DEFAULT PATHS =====================
ROOT            = Path(r"C:\SAIT\TennisAI")
DEFAULT_SOURCE  = str(ROOT / r"input_video\Video3_cut.mp4")
DEFAULT_MINIMAP = str(ROOT / r"Image\TennisCourtMap.jpeg")
DEFAULT_CALIB   = str(ROOT / "CALIB")
DEFAULT_OUT     = str(ROOT / r"outputs\Tennis_minimap_path_v12.mp4")
DETECT_RUNS     = ROOT / r"runs\detect"
# ========================================================

# ===================== UI / MINIMAP =====================
MINIMAP_POS   = "top-right"
MINIMAP_SCALE = 0.12
MARGIN_X      = 60
MARGIN_Y      = 120
OPACITY       = 0.70
USE_SHADOW    = True
WINDOW_SCALE  = 0.60  # scale for replay window
# ========================================================

PATH_COLOR  = (0, 255, 255)
PATH_THICK  = 2
DRAW_LAST   = 12
MAX_TRAIL   = 64
DOT_COLOR   = (0, 255, 0)

# ===================== BOUNCE SETTINGS =====================
BOUNCE_COLOR        = (0, 0, 255)
BOUNCE_RADIUS       = 6
BOUNCE_COOLDOWN_FR  = 4   # avoid double-trigger too close in time

BOUNCE_MM_MIN_DIST  = 18  # min distance between two red dots on minimap (px)

# Peak detector
DELTA_PIX = 0.35
CURV_MIN  = 0.40
LOW_PCT   = 10

# Classifier thresholds (kinematics)
SPEED_RATIO_BOUNCE_MAX = 1.30
SPEED_RATIO_HIT_MIN    = 1.40
THETA_MIN_HIT          = 20.0
# ========================================================

# ===================== Candidate (visual) =====================
CAND_SHOW_ON_MM   = False
CAND_COLOR        = (255, 200, 0)
CAND_THICK        = 2
CAND_R_BIG        = 8
CAND_R_MM         = 5
CAND_MM_COLOR     = (0, 0, 255)
CAND_HISTORY_LEN  = 60
CAND_MERGE_DIST   = 6
# ========================================================

# ===================== Anti false-bounce (static filter) ====
BOUNCE_COURT_MARGIN = 25
# ========================================================

RED   = (0,   0, 255)
GREEN = (0, 255,   0)
WHITE = (255,255,255)
BLACK = (0,   0,   0)

# ===================== InouT LOGO (replay only) =====================
LOGO_PATH          = Path(r"C:\SAIT\Capstone\Logo\InouTLogo.png")
LOGO_REL_HEIGHT    = 0.08   # logo height ~8% of video height
LOGO_MARGIN_RIGHT  = 90
LOGO_MARGIN_BOTTOM = 90
# ========================================================


# ===================== Utilities =====================
def draw_fps(img, fps_val):
    cv2.putText(img, f"{fps_val:.1f} FPS", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,BLACK,3,cv2.LINE_AA)
    cv2.putText(img, f"{fps_val:.1f} FPS", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,WHITE,1,cv2.LINE_AA)

def compute_center(xyxy):
    x1,y1,x2,y2 = xyxy
    return int((x1+x2)/2), int((y1+y2)/2)

def _read_json(p: Path):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def find_newest_model(runs_dir: Path) -> str | None:
    if not runs_dir.exists():
        return None
    pat   = re.compile(r"tennis_ball_v\d+$", re.IGNORECASE)
    cands = []
    for d in runs_dir.iterdir():
        if d.is_dir() and pat.match(d.name):
            best = d / "weights" / "best.pt"
            if best.exists():
                cands.append((best.stat().st_mtime, best))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return str(cands[0][1])

def _median3(pt_prev, pt_cur, pt_next):
    arr = [p for p in (pt_prev, pt_cur, pt_next) if p is not None]
    if len(arr) < 2:
        return pt_cur
    xs = sorted([p[0] for p in arr])
    ys = sorted([p[1] for p in arr])
    xm = xs[len(xs)//2]
    ym = ys[len(ys)//2]
    return (xm, ym)

def smooth_trail(points):
    n   = len(points)
    out = [None]*n
    for i in range(n):
        p = _median3(points[i-1] if i-1>=0 else None,
                     points[i],
                     points[i+1] if i+1<n else None)
        out[i] = p
    for i in range(1, n-1):
        if out[i-1] and out[i] and out[i+1]:
            out[i] = (out[i][0],
                      0.25*out[i-1][1] + 0.5*out[i][1] + 0.25*out[i+1][1])
    return out

# ===================== Calibration =====================
def load_calib(calib_dir: str, minimap_path: str):
    cdir = Path(calib_dir)
    H_f2c = None

    p = cdir / "H_frame_to_court_auto19.npy"
    if p.exists():
        try:
            H_f2c = np.load(str(p))
            print("✅ Loaded H: H_frame_to_court_auto19.npy")
        except Exception:
            H_f2c = None

    if H_f2c is None:
        cand = list(cdir.glob("H_frame_to_court_19pts.npy")) + \
               list(cdir.glob("H_frame_to_court_4pts.npy")) + \
               list(cdir.glob("H_frame_to_court*.npy"))
        if cand:
            cand.sort(key=lambda q: q.stat().st_mtime, reverse=True)
            try:
                H_f2c = np.load(str(cand[0]))
                print(f"✅ Loaded H: {cand[0].name}")
            except Exception:
                H_f2c = None

    if H_f2c is None:
        cand_cf = list(cdir.glob("H_court_to_frame_19pts.npy")) + \
                  list(cdir.glob("H_court_to_frame_4pts.npy")) + \
                  list(cdir.glob("H_court_to_frame*.npy"))
        if cand_cf:
            cand_cf.sort(key=lambda q: q.stat().st_mtime, reverse=True)
            try:
                H_c2f = np.load(str(cand_cf[0]))
                H_f2c = np.linalg.inv(H_c2f)
                print(f"✅ Loaded inv(H): {cand_cf[0].name}")
            except Exception:
                H_f2c = None

    court_size = None
    j = cdir / "anchors_tennis_19.json"
    if j.exists():
        meta = _read_json(j)
        sz   = meta.get("image_size")
        if isinstance(sz, dict) and "w" in sz and "h" in sz:
            court_size = (int(sz["w"]), int(sz["h"]))

    if court_size is None:
        jcand = list(cdir.glob("tennis_anchors_19pts.json")) + \
                list(cdir.glob("*anchors*.json")) + \
                list(cdir.glob("frame_points_19pts.json"))
        if jcand:
            jcand.sort(key=lambda q: q.stat().st_mtime, reverse=True)
            meta = _read_json(jcand[0])
            msz  = meta.get("image_size") or meta.get("court_size")
            if isinstance(msz, dict) and "w" in msz and "h" in msz:
                court_size = (int(msz["w"]), int(msz["h"]))

    if court_size is None:
        mm = cv2.imread(minimap_path)
        if mm is not None:
            court_size = (mm.shape[1], mm.shape[0])

    return H_f2c, court_size

# ===================== Minimap mapping =====================
def load_minimap_and_mapper(frame_shape, minimap_path, calib_dir, court_size):
    mm = cv2.imread(minimap_path)
    if mm is None or court_size is None:
        return None, None

    Hf, Wf = frame_shape[:2]
    mw = int(Wf * MINIMAP_SCALE)
    mh = int(mm.shape[0] * (mw / max(1, mm.shape[1])))
    mm_resized = cv2.resize(mm, (mw, mh), interpolation=cv2.INTER_AREA)

    j4 = Path(calib_dir) / "tennis_anchors_4pts.json"
    if not j4.exists():
        return mm_resized, None

    meta = _read_json(j4)
    pts4 = meta.get("points_px") or meta.get("image_points")
    if not pts4 or len(pts4) != 4:
        return mm_resized, None

    BL, BR, TR, TL = pts4
    pts_mm = np.array([TL, TR, BR, BL], dtype=np.float32)

    orig_w = meta["image_size"]["w"]
    orig_h = meta["image_size"]["h"]
    sx = mw / float(orig_w)
    sy = mh / float(orig_h)
    pts_mm[:, 0] *= sx
    pts_mm[:, 1] *= sy

    CW, CH = court_size
    src_court = np.array([[0,      0],
                          [CW-1,   0],
                          [CW-1, CH-1],
                          [0,    CH-1]], dtype=np.float32)

    H_c2mm, _ = cv2.findHomography(src_court, pts_mm, cv2.RANSAC, 3.0)
    return mm_resized, H_c2mm

def place_minimap_xy(W, H, mw, mh):
    if MINIMAP_POS == "top-left":
        return MARGIN_X, MARGIN_Y
    elif MINIMAP_POS == "top-right":
        return W - mw - MARGIN_X, MARGIN_Y
    elif MINIMAP_POS == "bottom-left":
        return MARGIN_X, H - mh - MARGIN_Y
    else:
        return W - mw - MARGIN_X, H - mh - MARGIN_Y

def add_minimap(frame,
                minimap,
                trail_points_court=None,
                court_size=None,
                H_c2mm=None,
                bounce_points_court=None,
                bounce_points_mm=None,
                cand_points_mm=None,
                cand_points_mm_hist=None):
    if minimap is None:
        return frame

    H, W = frame.shape[:2]
    mh, mw = minimap.shape[:2]
    if mw <= 0 or mh <= 0:
        return frame

    x1_des, y1_des = place_minimap_xy(W, H, mw, mh)
    x1 = max(0, min(W-1, x1_des))
    y1 = max(0, min(H-1, y1_des))
    roi_w = max(0, min(mw, W - x1))
    roi_h = max(0, min(mh, H - y1))
    if roi_w == 0 or roi_h == 0:
        return frame

    if USE_SHADOW:
        sh = frame.copy()
        cv2.rectangle(sh,
                      (x1+4, y1+4),
                      (x1+roi_w+4, y1+roi_h+4),
                      (0, 0, 0),
                      -1,
                      cv2.LINE_AA)
        frame[:] = cv2.addWeighted(sh, 0.28, frame, 0.72, 0)

    mm = minimap.copy()

    # Path (last DRAW_LAST points)
    if trail_points_court and court_size is not None:
        if H_c2mm is not None:
            pts = trail_points_court[:]

            def _transform_block(block):
                arr = np.array(block, np.float32).reshape(-1, 1, 2)
                out = cv2.perspectiveTransform(arr, H_c2mm).reshape(-1, 2)
                return [(int(round(x)), int(round(y))) for x, y in out]

            i = max(1, len(pts) - DRAW_LAST)
            while i < len(pts):
                if pts[i-1] is None or pts[i] is None:
                    i += 1
                    continue
                j = i - 1
                block = []
                while j < len(pts) and pts[j] is not None:
                    block.append(pts[j])
                    j += 1
                mm_pts = _transform_block(block)
                for k in range(1, len(mm_pts)):
                    cv2.line(mm, mm_pts[k-1], mm_pts[k],
                             PATH_COLOR, PATH_THICK, cv2.LINE_AA)
                i = j + 1

            last = next((p for p in reversed(pts) if p is not None), None)
            if last is not None:
                px, py = cv2.perspectiveTransform(
                    np.array([[last]], np.float32), H_c2mm
                )[0, 0]
                cv2.circle(mm,
                           (int(round(px)), int(round(py))),
                           5,
                           DOT_COLOR,
                           -1,
                           cv2.LINE_AA)
        else:
            cw, ch = court_size
            n = len(trail_points_court)
            start = max(1, n - DRAW_LAST + 1)
            for i in range(start, n):
                p1 = trail_points_court[i-1]
                p2 = trail_points_court[i]
                if p1 is None or p2 is None:
                    continue
                px1 = int((p1[0] / cw) * mw)
                py1 = int((p1[1] / ch) * mh)
                px2 = int((p2[0] / cw) * mw)
                py2 = int((p2[1] / ch) * mh)
                cv2.line(mm, (px1, py1), (px2, py2),
                         PATH_COLOR, PATH_THICK, cv2.LINE_AA)

            if trail_points_court[-1] is not None:
                px = int((trail_points_court[-1][0] / cw) * mw)
                py = int((trail_points_court[-1][1] / ch) * mh)
                cv2.circle(mm, (px, py), 5, DOT_COLOR, -1, cv2.LINE_AA)

    # Real bounces
    if bounce_points_mm:
        for (bx, by) in bounce_points_mm:
            cv2.circle(mm,
                       (int(bx), int(by)),
                       BOUNCE_RADIUS,
                       BOUNCE_COLOR,
                       -1,
                       cv2.LINE_AA)

    mm_roi   = mm[0:roi_h, 0:roi_w]
    base_roi = frame[y1:y1+roi_h, x1:x1+roi_w]
    if mm_roi.shape[:2] != base_roi.shape[:2]:
        mm_roi = cv2.resize(mm_roi,
                            (base_roi.shape[1], base_roi.shape[0]),
                            interpolation=cv2.INTER_AREA)

    frame[y1:y1+roi_h, x1:x1+roi_w] = cv2.addWeighted(
        base_roi, 1.0-OPACITY, mm_roi, OPACITY, 0
    )
    return frame

# ===================== Mapping & interpolation =====================
def frame_to_court_point(pt_xy, H_f2c, court_size, flip_y=True):
    if H_f2c is None or pt_xy is None or court_size is None:
        return None
    x, y = float(pt_xy[0]), float(pt_xy[1])
    dst  = cv2.perspectiveTransform(
        np.array([[[x, y]]], np.float32), H_f2c
    )[0, 0]
    Xc, Yc = float(dst[0]), float(dst[1])
    if flip_y:
        _, CH = court_size
        Yc = CH - Yc
    return (Xc, Yc)

def interpolate_missing_points(trail, max_gap=8):
    n = len(trail)
    out = trail[:]
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        start = i - 1
        j = i
        while j < n and out[j] is None:
            j += 1
        end = j
        gap = end - i
        if (start >= 0 and end < n
            and out[start] is not None
            and out[end]   is not None
            and gap <= max_gap):
            x1, y1 = out[start]
            x2, y2 = out[end]
            steps = gap + 1
            for k in range(1, steps):
                t = k / steps
                out[i + k - 1] = (x1 + t*(x2-x1), y1 + t*(y2-y1))
        i = end
    return out

# ===================== YOLO Snap ROI =====================
def yolo_detect_snap(model,
                     frame,
                     imgsz,
                     conf,
                     iou,
                     class_id=None,
                     last_xy=None,
                     roi_half=160,
                     roi_min=96,
                     conf_floor=0.25):
    H, W = frame.shape[:2]

    def _predict(img):
        kw = dict(source=img,
                  imgsz=imgsz,
                  conf=conf,
                  iou=iou,
                  verbose=False)
        try:
            if class_id is not None:
                kw["classes"] = [class_id]
        except Exception:
            pass
        r = model.predict(**kw)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return r, None
        confs = r.boxes.conf.cpu().numpy()
        best  = int(np.argmax(confs))
        if confs[best] < conf_floor:
            return r, None
        return r, best

    # Snap ROI around last_xy
    if last_xy is not None:
        cx, cy = map(int, last_xy)
        half   = int(max(roi_min, min(roi_half, 0.35*min(W, H))))
        x1     = max(0, cx-half)
        y1     = max(0, cy-half)
        x2     = min(W, cx+half)
        y2     = min(H, cy+half)
        crop   = frame[y1:y2, x1:x2]
        if crop.size > 0:
            r, idx = _predict(crop)
            if idx is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()[idx].astype(np.int32)
                return r, idx, np.array(
                    [xyxy[0]+x1, xyxy[1]+y1, xyxy[2]+x1, xyxy[3]+y1],
                    np.int32
                )

    # Fallback: full frame
    r, idx = _predict(frame)
    if idx is None:
        return None, None, None
    return r, idx, r.boxes.xyxy.cpu().numpy()[idx].astype(np.int32)

# ===================== Bounce helpers =====================
def _percentile(seq, p):
    if not seq:
        return None
    arr = np.array(sorted(seq), np.float32)
    k   = int(round((p/100.0)*(len(arr)-1)))
    k   = max(0, min(k, len(arr)-1))
    return float(arr[k])

def _median_vec(vecs):
    if not vecs:
        return np.array([0.0, 0.0], np.float32)
    arr = np.array(vecs, np.float32)
    return np.median(arr, axis=0)

def _angle_deg(a, b, eps=1e-6):
    a  = np.asarray(a, np.float32)
    b  = np.asarray(b, np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    cos = float(np.clip(np.dot(a, b)/(na*nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))

def find_peaks_y(trail_frame,
                 win=50,
                 prom_px=DELTA_PIX,
                 curv_min=CURV_MIN,
                 low_pct=LOW_PCT,
                 fps=30.0,
                 H=1080):
    n = len(trail_frame)
    L = min(win, n)
    if L < 7:
        return []
    idx0 = n - L
    buf  = trail_frame[idx0:]
    smooth = smooth_trail(buf)
    ys = [p[1] if p is not None else None for p in smooth]
    peaks = []
    for i in range(2, L-2):
        if ys[i-1] is None or ys[i] is None or ys[i+1] is None:
            continue
        if not (ys[i-1] < ys[i] > ys[i+1]):
            continue
        if (ys[i]-ys[i-1] < prom_px) or (ys[i]-ys[i+1] < prom_px):
            continue
        dy1 = ys[i]   - ys[i-1]
        dy2 = ys[i+1] - ys[i]
        if not (dy1 > 0 and dy2 < 0):
            continue
        xs  = np.arange(i-2, i+3, dtype=np.float32)
        y5  = np.array([ys[k] for k in range(i-2, i+3)], np.float32)
        a,b,c = np.polyfit(xs, y5, 2)
        if not (a < 0 and abs(a) >= curv_min):
            continue
        recent = [p[1] for p in buf[-int(max(10, fps*0.8)):] if p is not None]
        if recent:
            thr = _percentile(recent, low_pct)
            if ys[i] < thr:
                continue
        else:
            if ys[i] < 0.45*H:
                continue
        peaks.append(idx0 + i)
    dedup = []
    for pi in peaks:
        if not dedup or (pi - dedup[-1] >= 6):
            dedup.append(pi)
        else:
            yi_new = trail_frame[pi][1] if trail_frame[pi] else -1
            yi_old = trail_frame[dedup[-1]][1] if trail_frame[dedup[-1]] else -1
            if yi_new > yi_old:
                dedup[-1] = pi
    return dedup

def fallback_peaks_by_vy(trail_frame,
                         win=36,
                         min_drop_px=2.0):
    n = len(trail_frame)
    L = min(win, n)
    if L < 7:
        return []
    idx0 = n - L
    ys = [p[1] if p is not None else None for p in trail_frame[idx0:]]
    peaks = []
    for i in range(2, L-2):
        y0,y1,y2 = ys[i-2], ys[i-1], ys[i]
        if y0 is None or y1 is None or y2 is None:
            continue
        vy_prev = y1 - y0
        vy_now  = y2 - y1
        if vy_prev < -0.8 and vy_now > 0.2:
            if (y1 - min(y0, y2)) >= min_drop_px:
                peaks.append(idx0 + i - 1)
    uniq = []
    for pi in peaks:
        if not uniq or pi - uniq[-1] >= 5:
            uniq.append(pi)
    return uniq

# ---- Kinematics classifier (from v10) ----
def classify_event(trail_court,
                   trail_frame,
                   t0,
                   fps,
                   k=3,
                   r_bounce_max=SPEED_RATIO_BOUNCE_MAX,
                   r_hit_min=SPEED_RATIO_HIT_MIN,
                   theta_min_hit=THETA_MIN_HIT):
    """
    Returns:
        kind: 'bounce' | 'hit' | None
        pt_court: point at t0 (court-space, may be None)
        ratio, theta, sign_change (vy)
    """
    n = len(trail_court)
    if t0 < k or t0 + k >= n:
        return None, None, None, None, None

    def _slice(buf):
        pre  = [buf[i] for i in range(t0-k, t0+1)]
        post = [buf[i] for i in range(t0,   t0+k+1)]
        return pre, post

    preC, postC = _slice(trail_court)
    if any(p is None for p in preC) or any(p is None for p in postC):
        preC, postC = _slice(trail_frame)
        if any(p is None for p in preC) or any(p is None for p in postC):
            return None, None, None, None, None

    def _vels(seq):
        v = []
        for i in range(1, len(seq)):
            p1 = seq[i-1]
            p2 = seq[i]
            v.append(((p2[0]-p1[0])*fps,
                      (p2[1]-p1[1])*fps))
        return v

    vpre  = _vels(preC)
    vpost = _vels(postC)
    vp    = _median_vec(vpre)
    va    = _median_vec(vpost)
    sp    = float(np.linalg.norm(vp))
    sa    = float(np.linalg.norm(va))
    ratio = sa / max(sp, 1e-6)
    theta = _angle_deg(vp, va)
    sign_change = (vp[1] > 0 and va[1] < 0)

    pt_court = trail_court[t0] if trail_court[t0] is not None else None

    if sign_change and ratio <= r_bounce_max:
        return 'bounce', pt_court, ratio, theta, sign_change
    if ratio >= r_hit_min and theta >= theta_min_hit:
        return 'hit', pt_court, ratio, theta, sign_change

    return None, pt_court, ratio, theta, sign_change

# ---- parabola refinement ----
def refine_bounce_vertex(trail, t0, window=2):
    n = len(trail)
    l = max(0, t0-window)
    r = min(n-1, t0+window)
    ts = []
    ys = []
    for i in range(l, r+1):
        p = trail[i]
        if p is None:
            continue
        ts.append(float(i))
        ys.append(float(p[1]))
    if len(ts) < 5:
        return float(t0), trail[t0][1] if trail[t0] else None
    a,b,c = np.polyfit(np.array(ts, np.float32),
                       np.array(ys, np.float32), 2)
    if a >= 0:
        return float(t0), trail[t0][1] if trail[t0] else None
    tb = -b/(2*a)
    yb = a*tb*tb + b*tb + c
    return float(tb), float(yb)

# ---- STATIC FALSE-BOUNCE FILTER ----
def is_static_false_bounce(trail_frame,
                           t0,
                           window=3,
                           min_span=4,
                           min_dy=4.0,
                           min_path_len=5.0):
    n = len(trail_frame)
    if t0 < 0 or t0 >= n:
        return True
    l = max(0, t0 - window)
    r = min(n - 1, t0 + window)
    pts = [trail_frame[i] for i in range(l, r+1)
           if trail_frame[i] is not None]
    if len(pts) == 0:
        return True
    if len(pts) < min_span:
        return True
    ys = [p[1] for p in pts]
    dy = float(max(ys) - min(ys))
    total_path = 0.0
    for i in range(1, len(pts)):
        dx      = pts[i][0] - pts[i-1][0]
        dy_step = pts[i][1] - pts[i-1][1]
        total_path += (dx*dx + dy_step*dy_step) ** 0.5
    if dy < min_dy and total_path < min_path_len:
        return True
    return False

def court_contains(pt_court, court_size, margin=0):
    if pt_court is None or court_size is None:
        return False
    x, y = pt_court
    CW, CH = court_size
    return (-margin <= x <= CW-1+margin and
            -margin <= y <= CH-1+margin)

def is_new_minimap_bounce(mx, my, bounce_points_mm, min_dist=BOUNCE_MM_MIN_DIST):
    if not bounce_points_mm:
        return True
    for (ox, oy) in bounce_points_mm:
        dx = mx - ox
        dy = my - oy
        if dx*dx + dy*dy < (min_dist * min_dist):
            return False
    return True

# ---- Court-space kink detector ----
def find_court_kinks(trail_court,
                     win=40,
                     min_angle=30.0,
                     min_speed=1.2):
    n = len(trail_court)
    if n < 5:
        return []
    raw   = []
    start = max(1, n - win)
    for i in range(start, n-2):
        p0 = trail_court[i-1]
        p1 = trail_court[i]
        p2 = trail_court[i+1]
        if (p0 is None) or (p1 is None) or (p2 is None):
            continue
        v1 = np.array([p1[0]-p0[0], p1[1]-p0[1]], np.float32)
        v2 = np.array([p2[0]-p1[0], p2[1]-p1[1]], np.float32)
        s1 = float(np.linalg.norm(v1))
        s2 = float(np.linalg.norm(v2))
        if s1 < min_speed or s2 < min_speed:
            continue
        ang = _angle_deg(v1, v2)
        if ang >= min_angle:
            raw.append((i, ang))
    dedup = []
    for idx, ang in raw:
        if not dedup or idx - dedup[-1][0] >= 4:
            dedup.append([idx, ang])
        else:
            if ang > dedup[-1][1]:
                dedup[-1][0] = idx
                dedup[-1][1] = ang
    return [d[0] for d in dedup]

# ---- Merge close candidate indices ----
def merge_close_indices(idxs, min_sep=8):
    if not idxs:
        return []
    idxs   = sorted(idxs)
    merged = [idxs[0]]
    for idx in idxs[1:]:
        if idx - merged[-1] >= min_sep:
            merged.append(idx)
        else:
            merged[-1] = idx
    return merged

# ---- PARABOUNCE (fallback) ----
def parabounce_ok(trail_frame,
                  t0,
                  window=2,
                  min_dy=4.0,
                  a_thresh=-0.15):
    n = len(trail_frame)
    if t0 < 0 or t0 >= n:
        return False
    l = max(0, t0 - window)
    r = min(n - 1, t0 + window)
    xs = []
    ys = []
    for i in range(l, r + 1):
        p = trail_frame[i]
        if p is None:
            continue
        xs.append(float(i))
        ys.append(float(p[1]))
    if len(xs) < 5:
        return False
    xs_np = np.array(xs, np.float32)
    ys_np = np.array(ys, np.float32)
    a,b,c = np.polyfit(xs_np, ys_np, 2)
    if a >= a_thresh:
        return False
    dy = float(np.max(ys_np) - np.min(ys_np))
    if dy < min_dy:
        return False
    return True

def parabounce_hitlike(trail_frame,
                       t0,
                       fps,
                       k=3,
                       ratio_thresh=1.05,
                       theta_thresh=25.0):
    n = len(trail_frame)
    if t0 < k or t0 + k >= n:
        return False
    pre  = [trail_frame[i] for i in range(t0 - k, t0 + 1)]
    post = [trail_frame[i] for i in range(t0,     t0 + k + 1)]
    if any(p is None for p in pre) or any(p is None for p in post):
        return False

    def _vels(seq):
        v = []
        for i in range(1, len(seq)):
            x1,y1 = seq[i - 1]
            x2,y2 = seq[i]
            v.append(((x2 - x1) * fps,
                      (y2 - y1) * fps))
        return np.array(v, np.float32)

    vpre  = _vels(pre)
    vpost = _vels(post)
    if len(vpre) == 0 or len(vpost) == 0:
        return False
    vp    = np.median(vpre,  axis=0)
    va    = np.median(vpost, axis=0)
    sp    = float(np.linalg.norm(vp))
    sa    = float(np.linalg.norm(va))
    if sp < 1e-6:
        return False
    ratio = sa / sp
    theta = _angle_deg(vp, va)
    if (ratio >= ratio_thresh) and (theta >= theta_thresh):
        return True
    return False

# ===================== LOGO HELPERS (replay only) =====================
def load_and_prepare_logo(frame_shape):
    if not LOGO_PATH.exists():
        print(f"⚠️ Logo file not found: {LOGO_PATH}")
        return None
    logo = cv2.imread(str(LOGO_PATH), cv2.IMREAD_UNCHANGED)
    if logo is None:
        print("⚠️ Cannot load logo image.")
        return None

    Hf, Wf = frame_shape[:2]
    target_h = int(Hf * LOGO_REL_HEIGHT)
    if target_h <= 0:
        return None

    h0, w0 = logo.shape[:2]
    scale  = target_h / float(h0)
    new_w  = max(1, int(round(w0 * scale)))
    logo_resized = cv2.resize(logo, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return logo_resized

def draw_logo_on_frame(frame, logo_img):
    """
    Draw logo at bottom-right of frame.
    Returns:
        frame_out, (x1, y1, x2, y2) in full-frame coordinates.
    """
    if logo_img is None:
        return frame, None

    H, W = frame.shape[:2]
    lh, lw = logo_img.shape[:2]

    x2 = W - LOGO_MARGIN_RIGHT
    y2 = H - LOGO_MARGIN_BOTTOM
    x1 = max(0, x2 - lw)
    y1 = max(0, y2 - lh)

    roi_w = min(lw, W - x1)
    roi_h = min(lh, H - y1)
    if roi_w <= 0 or roi_h <= 0:
        return frame, None

    logo = logo_img[0:roi_h, 0:roi_w]

    if logo.shape[2] == 4:
        overlay = logo[:, :, :3]
        alpha   = logo[:, :, 3:].astype(np.float32) / 255.0
        roi     = frame[y1:y1+roi_h, x1:x1+roi_w].astype(np.float32)
        roi[:]  = overlay * alpha + roi * (1.0 - alpha)
        frame[y1:y1+roi_h, x1:x1+roi_w] = roi.astype(np.uint8)
    else:
        frame[y1:y1+roi_h, x1:x1+roi_w] = logo

    return frame, (x1, y1, x1+roi_w, y1+roi_h)

# ===================== TIMELINE (overlay) =====================
def draw_timeline(img,
                  frame_idx,
                  total_frames,
                  bounce_indices,
                  fps):
    if total_frames is None or total_frames <= 1:
        return img
    H, W = img.shape[:2]
    margin_x      = 80
    margin_bottom = 30
    bar_y  = H - margin_bottom
    bar_x1 = margin_x
    bar_x2 = W - margin_x
    bar_w  = max(1, bar_x2 - bar_x1)

    cv2.line(img,
             (bar_x1, bar_y),
             (bar_x2, bar_y),
             (160, 160, 160),
             2,
             cv2.LINE_AA)

    t_ratio = frame_idx / float(max(1, total_frames - 1))
    cur_x   = int(bar_x1 + t_ratio * bar_w)
    cv2.line(img,
             (cur_x, bar_y - 8),
             (cur_x, bar_y + 8),
             (0, 255, 255),
             2,
             cv2.LINE_AA)

    for bi in bounce_indices:
        if bi < 0 or bi >= total_frames:
            continue
        r  = bi / float(max(1, total_frames - 1))
        bx = int(bar_x1 + r * bar_w)
        cv2.circle(img,
                   (bx, bar_y),
                   4,
                   (0, 0, 255),
                   -1,
                   cv2.LINE_AA)

    cur_t   = frame_idx    / float(max(fps, 1e-6))
    total_t = total_frames / float(max(fps, 1e-6))
    txt = f"{cur_t:4.1f}s / {total_t:4.1f}s"
    cv2.putText(img,
                txt,
                (bar_x1, bar_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                BLACK,
                3,
                cv2.LINE_AA)
    cv2.putText(img,
                txt,
                (bar_x1, bar_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                WHITE,
                1,
                cv2.LINE_AA)
    return img

# ===================== INTERACTIVE REPLAY (seekable) =========
def interactive_replay(video_path: str,
                       fps: float,
                       bounce_events,
                       minimap,
                       court_size,
                       H_c2mm,
                       window_scale: float = WINDOW_SCALE,
                       logo_img=None):
    """
    bounce_events: list[{"frame": int, "court": (Xc, Yc) or None}]
    """
    cap2 = cv2.VideoCapture(video_path)
    if not cap2.isOpened():
        print(f"[⚠] Cannot open replay video: {video_path}")
        return

    total_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        print("[⚠] Replay video has no frames.")
        cap2.release()
        return

    win2 = "Replay 10s - InouT v12 (seek)"
    cv2.namedWindow(win2, cv2.WINDOW_NORMAL)

    state = {
        "idx": 0,
        "need_seek": True,
        "autoplay": True
    }

    bounce_frame_indices = [ev["frame"] for ev in bounce_events]

    # context for mouse callback
    mouse_ctx = {
        "logo_roi_view": None,
        "state": state,
        "bounce_events": bounce_events,
        "minimap": minimap,
        "court_size": court_size,
        "H_c2mm": H_c2mm,
        "fps": fps
    }

    def on_trackbar(pos):
        state["idx"]       = pos
        state["need_seek"] = True
        state["autoplay"]  = False

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ctx = param
        roi = ctx.get("logo_roi_view")
        if roi is None:
            return
        x1, y1, x2, y2 = roi
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return

        # CLICK on InouT logo
        idx   = ctx["state"]["idx"]
        evs   = ctx["bounce_events"]
        fps_l = ctx["fps"]

        if not evs:
            print("[InouT] No bounce events.")
            return

        # Find bounce closest to current frame
        best = None
        best_d = None
        for ev in evs:
            d = abs(ev["frame"] - idx)
            if best is None or d < best_d:
                best, best_d = ev, d

        if best is None:
            print("[InouT] No bounce found.")
            return

        # Only accept if close enough on timeline (~0.1s)
        max_delta = max(int(0.1 * fps_l), 3)
        if best_d > max_delta:
            print("[InouT] No bounce near current time.")
            return

        pt_court = best.get("court")
        if pt_court is None:
            print("[InouT] Bounce has no court position.")
            return

        trail_court = best.get("trail_court")

        print(f"[InouT] Opening decision view for frame {best['frame']} (Δ={best_d} frames).")
        try:
            inout_decision.inout_decision(
                ctx["minimap"],
                ctx["court_size"],
                ctx["H_c2mm"],
                [pt_court],               # 1 bounce
                trail_court=trail_court,  # ~3s history
                fps=fps_l
            )
        except Exception as e:
            print("[⚠] Error calling inout_decision:", e)
    

    cv2.createTrackbar("Timeline",
                       win2,
                       0,
                       max(total_frames - 1, 1),
                       on_trackbar)
    cv2.setMouseCallback(win2, on_mouse, mouse_ctx)

    print("🎾 Replay ready:")
    print(" SPACE = Pause/Play, ←/→ = step, ESC/Q = quit, drag slider to seek (auto-stop).")
    print(" CLICK InouT logo (bottom-right) to open In/Out Decision.")

    logo_bbox_frame = None

    while True:
        if state["autoplay"] and not state["need_seek"]:
            if state["idx"] < total_frames - 1:
                state["idx"] += 1
                state["need_seek"] = True
            else:
                state["autoplay"] = False

        if state["need_seek"]:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, state["idx"])
            ok, frame = cap2.read()
            if not ok:
                break

            frame_disp = frame.copy()
            frame_disp = draw_timeline(
                frame_disp,
                frame_idx=state["idx"],
                total_frames=total_frames,
                bounce_indices=bounce_frame_indices,
                fps=fps
            )

            frame_disp, logo_bbox_frame = draw_logo_on_frame(frame_disp, logo_img)

            view = cv2.resize(
                frame_disp,
                (int(frame_disp.shape[1] * window_scale),
                 int(frame_disp.shape[0] * window_scale))
            )

            # scale logo ROI into view coordinates
            if logo_bbox_frame is not None:
                x1, y1, x2, y2 = logo_bbox_frame
                sx = window_scale
                sy = window_scale
                mouse_ctx["logo_roi_view"] = (
                    int(x1 * sx), int(y1 * sy),
                    int(x2 * sx), int(y2 * sy)
                )
            else:
                mouse_ctx["logo_roi_view"] = None

            cv2.imshow(win2, view)
            cv2.setTrackbarPos("Timeline", win2, state["idx"])
            state["need_seek"] = False

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):  # ESC / q
            break
        elif key == ord(' '):
            state["autoplay"] = not state["autoplay"]
        elif key == 81:            # ←
            state["autoplay"]  = False
            state["idx"]       = max(0, state["idx"] - 1)
            state["need_seek"] = True
        elif key == 83:            # →
            state["autoplay"]  = False
            state["idx"]       = min(total_frames - 1, state["idx"] + 1)
            state["need_seek"] = True

    cap2.release()
    cv2.destroyWindow(win2)

# ============================= MAIN =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default=None,
                    help="Path to YOLO .pt; if omitted, auto-pick newest tennis_ball_v*")
    ap.add_argument("--source",  default=DEFAULT_SOURCE)
    ap.add_argument("--minimap", default=DEFAULT_MINIMAP)
    ap.add_argument("--calib",   default=DEFAULT_CALIB)
    ap.add_argument("--out",     default=DEFAULT_OUT)
    ap.add_argument("--show",    action="store_true", default=False)
    ap.add_argument("--class_id", type=int, default=None)
    ap.add_argument("--flip_y",   type=int, default=1)
    args = ap.parse_args()

    model_path = args.model or find_newest_model(DETECT_RUNS)
    if not model_path:
        sys.exit("❌ Cannot find model. Provide --model or ensure training runs exist.")
    print(f"🧠 Using model: {model_path}")

    cap = cv2.VideoCapture(args.source)
    ok0, f0 = cap.read()
    if not ok0:
        sys.exit(f"❌ Cannot read video: {args.source}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    W, H = f0.shape[1], f0.shape[0]
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    H_f2c, court_size = load_calib(args.calib, args.minimap)
    minimap, H_c2mm   = load_minimap_and_mapper(
        f0.shape, args.minimap, args.calib, court_size
    )
    if H_f2c is None or court_size is None:
        print("⚠️ Missing calibration (H_f2c or court_size).")
    if minimap is None:
        print("⚠️ Minimap not available; overlay disabled.")

    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )
    model = YOLO(model_path)

    last_xy             = None
    trail_court         = []
    trail_frame         = []
    bounce_points_court = []
    bounce_points_mm    = []
    bounce_cooldown     = 0
    cand_mm_history     = []

    fps_smooth = 0
    t_prev     = time.time()

    # Full list of bounce events (frame + court point)
    bounce_events = []

    win = "Tennis Ball Track v12"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("▶ Detect v12 is running on the clip (headless or with --show)...")

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()

        res, idx, xyxy = yolo_detect_snap(
            model,
            frame,
            imgsz=960,
            conf=0.25,
            iou=0.45,
            class_id=args.class_id,
            last_xy=last_xy,
            roi_half=160,
            roi_min=96,
            conf_floor=0.20
        )

        annotated = frame.copy()
        dot       = None

        if res is not None and idx is not None and xyxy is not None:
            cx, cy = compute_center(xyxy)
            dot = (cx, cy)

        trail_frame.append(dot if dot is not None else None)
        if len(trail_frame) > MAX_TRAIL:
            trail_frame.pop(0)

        if dot is not None:
            last_xy = dot
            if H_f2c is not None and court_size is not None:
                cpt = frame_to_court_point(
                    dot, H_f2c, court_size, flip_y=bool(args.flip_y)
                )
                trail_court.append(cpt)
            else:
                trail_court.append(None)
        else:
            trail_court.append(None)

        if len(trail_court) > MAX_TRAIL:
            trail_court.pop(0)

        trail_court = interpolate_missing_points(trail_court, max_gap=8)

        peak_list = find_peaks_y(
            trail_frame,
            win=50,
            prom_px=DELTA_PIX,
            curv_min=CURV_MIN,
            low_pct=LOW_PCT,
            fps=fps,
            H=H
        )
        fallback_list = fallback_peaks_by_vy(
            trail_frame,
            win=36,
            min_drop_px=2.0
        )
        all_peaks = sorted(set(peak_list) | set(fallback_list))

        kink_list = find_court_kinks(
            trail_court,
            win=40,
            min_angle=30.0,
            min_speed=1.2
        )
        all_peaks = sorted(set(all_peaks) | set(kink_list))
        all_peaks = merge_close_indices(all_peaks, min_sep=8)

        cand_points_mm_frame = []
        if CAND_SHOW_ON_MM and H_c2mm is not None and minimap is not None:
            for peak_idx in all_peaks:
                if (0 <= peak_idx < len(trail_court)
                    and trail_court[peak_idx] is not None):
                    arr = np.array(
                        [[[trail_court[peak_idx][0],
                           trail_court[peak_idx][1]]]], dtype=np.float32
                    )
                    try:
                        mx, my = cv2.perspectiveTransform(arr, H_c2mm)[0, 0]
                        if np.isfinite(mx) and np.isfinite(my):
                            cand_points_mm_frame.append(
                                (int(np.clip(round(mx), 0, minimap.shape[1]-1)),
                                 int(np.clip(round(my), 0, minimap.shape[0]-1)))
                            )
                    except Exception:
                        pass

        for peak_idx in all_peaks:
            if 0 <= peak_idx < len(trail_frame) and trail_frame[peak_idx] is not None:
                px, py = int(trail_frame[peak_idx][0]), int(trail_frame[peak_idx][1])
                cv2.circle(annotated,
                           (px, py),
                           CAND_R_BIG,
                           CAND_COLOR,
                           CAND_THICK,
                           cv2.LINE_AA)

        def _merge_into_history(history, new_pts, max_len, merge_dist):
            for (nx, ny) in new_pts:
                merged = False
                for i, (hx, hy) in enumerate(history):
                    if (abs(hx - nx) <= merge_dist) and (abs(hy - ny) <= merge_dist):
                        history[i] = (nx, ny)
                        merged = True
                        break
                if not merged:
                    history.append((nx, ny))
            if len(history) > max_len:
                del history[:-max_len]

        if CAND_SHOW_ON_MM:
            _merge_into_history(
                cand_mm_history,
                cand_points_mm_frame,
                CAND_HISTORY_LEN,
                CAND_MERGE_DIST
            )

        if bounce_cooldown > 0:
            bounce_cooldown -= 1

        # ================== BOUNCE ANALYSIS ==================
        for peak_idx in all_peaks:
            if bounce_cooldown > 0:
                break
            t0 = peak_idx

            if is_static_false_bounce(
                trail_frame, t0,
                window=3,
                min_span=4,
                min_dy=4.0,
                min_path_len=5.0
            ):
                continue

            kind, pt_court, ratio, theta, sign_change = classify_event(
                trail_court, trail_frame, t0, fps, k=3
            )
            accepted = False
            if kind == 'bounce':
                accepted = True
            else:
                if (parabounce_ok(trail_frame,
                                  t0,
                                  window=2,
                                  min_dy=4.0,
                                  a_thresh=-0.15)
                    and not parabounce_hitlike(trail_frame,
                                               t0,
                                               fps,
                                               k=3,
                                               ratio_thresh=1.05,
                                               theta_thresh=25.0)):
                    accepted = True

            if (pt_court is None) and (0 <= t0 < len(trail_court)):
                pt_court = trail_court[t0]

            if not accepted:
                continue

            tb, yb = refine_bounce_vertex(trail_frame, t0, window=2)
            t_int = int(np.clip(round(tb), 0, len(trail_frame) - 1))

            pt_frame = None
            if 0 <= t_int < len(trail_frame):
                pt_frame = trail_frame[t_int]
            elif 0 <= t0 < len(trail_frame):
                pt_frame = trail_frame[t0]

            pt_court_ref = None
            if 0 <= t_int < len(trail_court):
                pt_court_ref = trail_court[t_int]
            elif 0 <= t0 < len(trail_court):
                pt_court_ref = trail_court[t0]
            else:
                pt_court_ref = pt_court

            new_bounce = True
            mx = my = None
            if (pt_court_ref is not None) and (H_c2mm is not None) and (minimap is not None):
                arr = np.array(
                    [[[pt_court_ref[0], pt_court_ref[1]]]],
                    dtype=np.float32
                )
                try:
                    mx_f, my_f = cv2.perspectiveTransform(arr, H_c2mm)[0, 0]
                    if np.isfinite(mx_f) and np.isfinite(my_f):
                        mx = int(np.clip(round(mx_f), 0, minimap.shape[1]-1))
                        my = int(np.clip(round(my_f), 0, minimap.shape[0]-1))
                        new_bounce = is_new_minimap_bounce(
                            mx, my, bounce_points_mm, BOUNCE_MM_MIN_DIST
                        )
                    else:
                        new_bounce = False
                except Exception:
                    new_bounce = False

            if (H_c2mm is not None and minimap is not None) and not new_bounce:
                continue

            bounce_cooldown = BOUNCE_COOLDOWN_FR

            if pt_frame is not None:
                bx, by = int(pt_frame[0]), int(pt_frame[1])
                cv2.circle(annotated,
                           (bx, by),
                           BOUNCE_RADIUS,
                           BOUNCE_COLOR,
                           -1,
                           cv2.LINE_AA)

            bounce_points_court.append(pt_court_ref)
            if mx is not None and my is not None:
                bounce_points_mm.append((mx, my))

            # save ~3s of history for InouT decision minimap replay
            trail_len = len(trail_court)
            if fps > 0:
                desired_len = int(fps * 3.0)
            else:
                desired_len = 75
            history_len = min(trail_len, desired_len)
            trail_slice = trail_court[trail_len - history_len : trail_len]

            bounce_events.append({
                "frame": frame_idx,
                "court": pt_court_ref,
                "trail_court": trail_slice
            })


        annotated = add_minimap(
            annotated,
            minimap,
            trail_points_court=trail_court,
            court_size=court_size,
            H_c2mm=H_c2mm,
            bounce_points_court=bounce_points_court,
            bounce_points_mm=bounce_points_mm,
            cand_points_mm=None,
            cand_points_mm_hist=None
        )

        if dot is not None:
            cv2.circle(annotated, dot, 5, GREEN, -1, cv2.LINE_AA)

        fps_cur = 1.0 / max(1e-6, now - t_prev)
        t_prev  = now
        fps_smooth = 0.9*fps_smooth + 0.1*fps_cur
        draw_fps(annotated, fps_smooth)

        writer.write(annotated)

        if args.show:
            view = cv2.resize(
                annotated,
                (int(W * WINDOW_SCALE),
                 int(H * WINDOW_SCALE))
            )
            cv2.imshow(win, view)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyWindow(win)

    print("✅ Saved:", args.out)

    # Prepare logo (based on first frame)
    logo_img = load_and_prepare_logo(f0.shape)

    # Interactive player with timeline + logo + click-to-decision
    try:
        interactive_replay(
            args.out,
            fps,
            bounce_events,
            minimap,
            court_size,
            H_c2mm,
            WINDOW_SCALE,
            logo_img
        )
    except Exception as e:
        print("[⚠] Interactive replay failed:", e)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
