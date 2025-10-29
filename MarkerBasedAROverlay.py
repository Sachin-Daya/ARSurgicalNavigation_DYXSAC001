# aruco_overlay_single_image_filled_dual.py
# Overlay SKULL + LESION + MARKERS (filled) onto a single image.
# Selects the detected ArUco marker closest to the camera by default
# (set TOP_ID >= 0 to force a specific marker).
#
# On-screen controls:
#   J/L : translate X ±2 mm      U/O : translate Y ±2 mm      I/K : translate Z ±2 mm (Z<0 below tag plane)
#   1/2 : rotate Rx ±1°          3/4 : rotate Ry ±1°          5/6 : rotate Rz ±1°
#   9/0 : scale all meshes ×/÷ 1.01
#   F   : toggle filled / wireframe (starts filled)
#   S   : write T_marker_skull.json (current marker→skull transform)
#   P   : save overlay_result.png (screenshot)
#   R   : reset tx,ty,tz,rx,ry,rz,scale to defaults
#   H   : toggle help overlay
#   Q/ESC: quit

import cv2, json, math, numpy as np
from pathlib import Path
import time  # runtime metrics

# ========= CONFIG =========
IMAGE_PATH       = "Image13.jpg"                 # source image
NPZ_PATH         = "calibration_data_ipad.npz"   # intrinsics of the camera used for IMAGE_PATH

SKULL_OBJ_PATH   = "skull.obj"
LESION_OBJ_PATH  = "lesion.obj"                  # optional; render if present
MARKERS_OBJ_PATH = "Markers.obj"                 # optional; render if present

TOP_ID           = -1                            # <0: auto-pick nearest marker; >=0: force this ID if detected
MARKER_SIZE_M    = 0.053                         # black-square side length in meters

T_JSON_PATH      = "T_marker_skull.json"         # persisted marker→skull transform (if exists, loads on start)

# Initial transform when no JSON exists (marker frame; tz<0 positions the model under the tag plane).
TX, TY, TZ       = 0.000, 0.000, -0.060          # meters
RX, RY, RZ       = 90.0, 0.0, 0.0                # degrees; +90° about X for Blender +Z-up convention

# Per-mesh unit scales (e.g., OBJs authored in millimetres → 0.001).
OBJ_SCALE_SKULL   = 0.001
OBJ_SCALE_LESION  = 0.001
OBJ_SCALE_MARKERS = 0.001

# Render styling
DRAW_FILLED      = True                          # start with filled triangles
COLOR_SKULL      = (255, 255, 255)               # BGR
COLOR_LESION     = (0, 255, 0)
COLOR_MARKERS    = (0, 0, 255)
SKULL_ALPHA      = 0.6                           # skull opacity for compositing
Z_NEAR_M         = 0.02                          # near-plane cull (meters)
# =========================

def euler_xyz_deg(rx, ry, rz):
    # Compose rotation Rz * Ry * Rx from degrees.
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx); cy, sy = np.cos(ry), np.sin(ry); cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

def make_T(R=None, t=None):
    # Build a 4×4 homogeneous transform from rotation and translation.
    T = np.eye(4, dtype=np.float64)
    if R is not None: T[:3,:3] = R
    if t is not None: T[:3, 3] = np.asarray(t).reshape(3)
    return T

def rvec_tvec_to_T(rvec, tvec):
    # Convert Rodrigues rvec/tvec to a homogeneous transform.
    R, _ = cv2.Rodrigues(rvec); return make_T(R, tvec)

def load_calib(npz_path):
    # Load intrinsics and distortion (and optional image size) from NPZ.
    d = np.load(npz_path, allow_pickle=True)
    return d["K"].astype(np.float32), d["dist"].astype(np.float32), tuple(d.get("image_size", (0,0)))

def scale_K(K_in, size_calib, w_img, h_img):
    # Heuristic: scale K to the current image resolution, accounting for (w,h) vs (h,w) mismatches.
    w0, h0 = (int(size_calib[0]), int(size_calib[1])) if size_calib != (0,0) else (w_img, h_img)
    candidates = [(w0,h0), (h0,w0), (w_img,h_img)]
    best = None; best_err = 1e9
    for wc,hc in candidates:
        sx, sy = w_img/float(wc), h_img/float(hc)
        K = K_in.copy()
        K[0,0] *= sx; K[0,2] *= sx; K[1,1] *= sy; K[1,2] *= sy
        err = abs(w_img - wc) + abs(h_img - hc)
        if err < best_err: best_err, best = err, K
    return best.astype(np.float32)

def load_obj_vertices_faces(path):
    # Read OBJ as (V,F). Prefer Open3D if available; fallback to a minimal parser that triangulates quads.
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(path))
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.triangles, dtype=np.int32)
        if V.size and F.size: return V, F
        raise RuntimeError
    except Exception:
        V, F = [], []
        with open(path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    _, x, y, z = line.strip().split()[:4]
                    V.append([float(x), float(y), float(z)])
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    idxs = [int(p.split("/")[0]) - 1 for p in parts]
                    if len(idxs) == 3: F.append(idxs)
                    elif len(idxs) == 4:
                        F.append([idxs[0], idxs[1], idxs[2]])
                        F.append([idxs[0], idxs[2], idxs[3]])
        if not V or not F: raise RuntimeError(f"Failed to load OBJ: {path}")
        return np.array(V, dtype=np.float64), np.array(F, dtype=np.int32)

def project_cam_points(K, dist, pts_cam):
    # Project camera-frame 3D points using current intrinsics/distortion.
    imgpts, _ = cv2.projectPoints(pts_cam, np.zeros((3,1)), np.zeros((3,1)), K, dist)
    return imgpts.reshape(-1,2)

def sort_faces_back_to_front(V_cam, F):
    # Painter’s algorithm: farthest faces first by average Z in camera frame.
    return F[np.argsort(V_cam[F].mean(axis=1)[:,2])]

def draw_mesh_filled(frame, V_cam, imgpts, F, face_color=(0,255,0), alpha=0.28):
    # Triangle fill with simple back-to-front compositing and degenerate/oversized polygon rejection.
    h, w = frame.shape[:2]
    F_sorted = sort_faces_back_to_front(V_cam, F)
    overlay = frame.copy()
    img_area = float(w*h)
    for tri in F_sorted:
        poly = imgpts[tri].astype(np.int32)
        if not np.isfinite(poly).all(): continue
        area = cv2.contourArea(poly.astype(np.float32))
        if area <= 0.0 or area > 0.25*img_area:
            continue
        cv2.fillConvexPoly(overlay, poly, face_color)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

# --- Help overlay ---
def draw_help(frame, lines, org=(10,10), font_scale=0.55, line_spacing_px=22, bg_alpha=0.6):
    # Render a semi-transparent help panel with monospaced alignment.
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1
    text_color = (240, 240, 240)
    shadow_color = (20, 20, 20)
    bg_color = (0, 0, 0)
    border_color = (100, 100, 100)
    max_w = 0
    for s in lines:
        (w, h), _ = cv2.getTextSize(s, font, font_scale, font_thickness)
        max_w = max(max_w, w)
    box_w = max_w + 16
    box_h = int(line_spacing_px * len(lines) + 12)
    x1, y1 = x - 6, y - 6
    x2, y2 = x + box_w, y + box_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, 1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    yy = y + 10
    for s in lines:
        cv2.putText(frame, s, (x+1, yy+1), font, font_scale, shadow_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, s, (x, yy), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        yy += line_spacing_px

# --- Marker picking ---
def pick_nearest_marker_by_pose(img_gray, corners, ids, K, dist, marker_size_m, prefer_id):
    """
    If prefer_id >= 0 and present, return that marker.
    Otherwise, estimate pose for each detection and choose the one with the smallest ||tvec||.
    Returns: corners, id, rvec, tvec, distance
    """
    if ids is None or len(ids) == 0:
        return None, None, None, None, None

    ids_r = ids.ravel()

    # Forced selection
    if prefer_id is not None and prefer_id >= 0:
        match = np.where(ids_r == prefer_id)[0]
        if len(match):
            idx = int(match[0])
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[idx], marker_size_m, K, dist)
            rvec, tvec = rvec[0,0], tvec[0,0]
            dist_m = float(np.linalg.norm(tvec))
            return corners[idx], int(ids_r[idx]), rvec, tvec, dist_m
        # fall back to nearest if forced ID not found

    # Auto-pick nearest
    best = (1e9, None, None, None, None)  # dist, corners, id, rvec, tvec
    for idx, cid in enumerate(ids_r):
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[idx], marker_size_m, K, dist)
        rvec, tvec = rvec[0,0], tvec[0,0]
        dist_m = float(np.linalg.norm(tvec))
        if dist_m < best[0]:
            best = (dist_m, corners[idx], int(cid), rvec, tvec)
    dist_m, c_best, id_best, rvec_best, tvec_best = best
    return c_best, id_best, rvec_best, tvec_best, dist_m

def main():
    global TX, TY, TZ, RX, RY, RZ, OBJ_SCALE_SKULL, OBJ_SCALE_LESION, OBJ_SCALE_MARKERS, DRAW_FILLED

    startup_timer_start = time.time()

    # Load image and camera intrinsics (scale K if image size differs from calibration size).
    img = cv2.imread(IMAGE_PATH)
    if img is None: raise FileNotFoundError(IMAGE_PATH)
    h, w = img.shape[:2]
    K0, dist0, size0 = load_calib(NPZ_PATH)
    K = scale_K(K0, size0, w, h)
    dist = dist0

    # Load meshes (skull required; lesion/markers optional).
    V_skl_raw, F_skl = load_obj_vertices_faces(SKULL_OBJ_PATH)
    V_les_raw, F_les = load_obj_vertices_faces(LESION_OBJ_PATH) if Path(LESION_OBJ_PATH).exists() else (None, None)
    V_mrk_raw, F_mrk = load_obj_vertices_faces(MARKERS_OBJ_PATH) if Path(MARKERS_OBJ_PATH).exists() else (None, None)

    startup_time = (time.time() - startup_timer_start) * 1000
    print(f"--- Metrics: Asset loading complete in {startup_time:.2f} ms ---")

    # ArUco detection (OpenCV >= 4.7 uses ArucoDetector; fallback for older versions).
    detect_timer_start = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _rej = detector.detectMarkers(gray)
    except AttributeError:
        corners, ids, _rej = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    detect_time = (time.time() - detect_timer_start) * 1000
    print(f"--- Metrics: Marker detection complete in {detect_time:.2f} ms ---")

    if ids is None or len(ids) == 0:
        cv2.imshow("Image", img)
        cv2.displayOverlay("Image", f"No markers found.", 3000)
        cv2.waitKey(0); return

    # Select marker: forced ID if present, else nearest by estimated pose.
    c, picked_id, rvec, tvec, picked_dist = pick_nearest_marker_by_pose(
        gray, corners, ids, K, dist, MARKER_SIZE_M, prefer_id=TOP_ID
    )
    if c is None:
        cv2.imshow("Image", img)
        cv2.displayOverlay("Image", f"No marker (or TOP_ID {TOP_ID}) found.", 3000)
        cv2.waitKey(0); return

    T_cam_marker = rvec_tvec_to_T(rvec, tvec)

    # Reprojection RMSE for the chosen marker (sanity check).
    M = MARKER_SIZE_M / 2.0
    marker_3d_corners = np.array([
        [-M,  M, 0],
        [ M,  M, 0],
        [ M, -M, 0],
        [-M, -M, 0]
    ], dtype=np.float32)
    reprojected_2d_corners, _ = cv2.projectPoints(marker_3d_corners, rvec, tvec, K, dist)
    original_2d_corners = c.reshape(4, 2)
    err = reprojected_2d_corners.reshape(4, 2) - original_2d_corners
    error_per_corner = np.linalg.norm(err, axis=1)
    reprojection_error_rmse = float(np.sqrt(np.mean(error_per_corner**2)))
    print(f"--- Metrics: Chosen marker ID={picked_id}, dist={picked_dist:.3f} m, reprojection RMSE={reprojection_error_rmse:.3f} px ---")

    # Load existing marker→skull transform if present; otherwise, use the live defaults.
    T_marker_skull_from_file = None
    if Path(T_JSON_PATH).exists():
        print(f"Loading initial transform from {T_JSON_PATH}")
        T_marker_skull_from_file = np.array(json.load(open(T_JSON_PATH)))
        TX, TY, TZ = T_marker_skull_from_file[0:3, 3]

    using_live_vars = (T_marker_skull_from_file is None)
    if not using_live_vars:
        print("File loaded. Using file’s T matrix. Press any transform key to start editing live values.")
    else:
        print("No file found. Starting with default TX, TY, TZ, RX, RY, RZ.")

    # UI loop
    win = "AR Overlay (image) — press H for help"
    show_help = True
    adjustment_key_presses = 0

    HELP_LINES = [
        "--- CONTROLS ---",
        "Translate (2mm):",
        "  J/L: X-Axis (Left/Right)",
        "  U/O: Y-Axis (Up/Down)",
        "  I/K: Z-Axis (In/Out)",
        "",
        "Rotate (1°):",
        "  1/2: X-Axis (Pitch)",
        "  3/4: Y-Axis (Yaw)",
        "  5/6: Z-Axis (Roll)",
        "",
        "Scale (1.01):",
        "  9/0: Scale All Meshes",
        "",
        "--- SYSTEM ---",
        "  F: Toggle Fill / Wireframe",
        "  S: Save Transform (T_marker_skull.json)",
        "  P: Save Screenshot (overlay_result.png)",
        "  R: Reset All Transforms",
        "  H: Toggle This Help",
        "  Q/ESC: Quit"
    ]

    while True:
        redraw_timer_start = time.time()
        frame = img.copy()

        # Choose transform source (loaded file vs live editable variables).
        if using_live_vars:
            R_align = euler_xyz_deg(RX, RY, RZ)
            Tms = make_T(R_align, [TX, TY, TZ])
        else:
            Tms = T_marker_skull_from_file

        T_cam_skull = T_cam_marker @ Tms

        # Scale vertices to meters (if authored in other units).
        V_skl = (V_skl_raw * OBJ_SCALE_SKULL).astype(np.float64)
        if V_les_raw is not None: V_les = (V_les_raw * OBJ_SCALE_LESION).astype(np.float64)
        if V_mrk_raw is not None: V_mrk = (V_mrk_raw * OBJ_SCALE_MARKERS).astype(np.float64)

        # Transform to camera frame.
        def xform(V):
            Vh = np.c_[V, np.ones(len(V))]
            return (T_cam_skull @ Vh.T).T[:, :3]

        V_skl_cam = xform(V_skl)
        if V_les_raw is not None: V_les_cam = xform(V_les)
        if V_mrk_raw is not None: V_mrk_cam = xform(V_mrk)

        # Face culling: drop triangles behind the near plane or with non-finite vertices.
        def filter_faces(V_cam, F):
            if V_cam is None or F is None: return None, None
            z = V_cam[:,2]
            finite = np.isfinite(V_cam).all(axis=1)
            ok = np.all((z[F] > Z_NEAR_M), axis=1) & np.all(finite[F], axis=1)
            F_use = F[ok]
            if len(F_use)==0: return V_cam, F_use
            return V_cam, F_use

        V_skl_cam, F_skl_use = filter_faces(V_skl_cam, F_skl)
        if V_les_raw is not None: V_les_cam, F_les_use = filter_faces(V_les_cam, F_les)
        if V_mrk_raw is not None: V_mrk_cam, F_mrk_use = filter_faces(V_mrk_cam, F_mrk)

        # Project to image coordinates.
        def project(V_cam):
            return project_cam_points(K, dist, V_cam)

        if len(F_skl_use): uv_skl = project(V_skl_cam)
        if V_les_raw is not None and len(F_les_use): uv_les = project(V_les_cam)
        if V_mrk_raw is not None and len(F_mrk_use): uv_mrk = project(V_mrk_cam)

        # Draw with chosen style.
        if DRAW_FILLED:
            if V_les_raw is not None and len(F_les_use):
                draw_mesh_filled(frame, V_les_cam, uv_les, F_les_use, face_color=COLOR_LESION, alpha=1.0)
            if V_mrk_raw is not None and len(F_mrk_use):
                draw_mesh_filled(frame, V_mrk_cam, uv_mrk, F_mrk_use, face_color=COLOR_MARKERS, alpha=1.0)
            if len(F_skl_use):
                draw_mesh_filled(frame, V_skl_cam, uv_skl, F_skl_use, face_color=COLOR_SKULL, alpha=SKULL_ALPHA)
        else:
            def draw_wire(uv, F, color):
                h2, w2 = frame.shape[:2]
                for tri in F:
                    poly = uv[tri].astype(np.int32)
                    if np.any((poly[:,0]<-50)|(poly[:,0]>w2+50)|(poly[:,1]<-50)|(poly[:,1]>h2+50)): continue
                    cv2.polylines(frame, [poly], True, color, 1, cv2.LINE_AA)
            if V_les_raw is not None and len(F_les_use): draw_wire(uv_les, F_les_use, COLOR_LESION)
            if V_mrk_raw is not None and len(F_mrk_use): draw_wire(uv_mrk, F_mrk_use, COLOR_MARKERS)
            if len(F_skl_use): draw_wire(uv_skl, F_skl_use, (200,200,200))

        # Visualize chosen marker and its axes.
        cv2.aruco.drawDetectedMarkers(frame, [c], np.array([[picked_id]], dtype=np.int32))
        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, MARKER_SIZE_M*0.5)

        # HUD with current distances, scales, and transform source.
        hud = (f"ID:{picked_id}  dist={picked_dist:.3f}m  "
               f"scale_skl={OBJ_SCALE_SKULL:.4f} scale_les={OBJ_SCALE_LESION:.4f} "
               f"scale_mrk={OBJ_SCALE_MARKERS:.4f}")
        if not using_live_vars:
            hud_tx, hud_ty, hud_tz = T_marker_skull_from_file[0:3, 3]
            hud += f"  tx={hud_tx:.3f} ty={hud_ty:.3f} tz={hud_tz:.3f}  (Loaded from JSON)"
            hud_color = (180, 180, 180)
        else:
            hud += f"  tx={TX:.3f} ty={TY:.3f} tz={TZ:.3f}  rx={RX:.1f} ry={RY:.1f} rz={RZ:.1f}"
            hud_color = (30, 230, 30)

        cv2.putText(frame, hud, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2, cv2.LINE_AA)

        # Metrics: reprojection error, number of adjustments, and redraw time.
        redraw_time = (time.time() - redraw_timer_start) * 1000
        metric_text = (f"Reprojection Err: {reprojection_error_rmse:.3f} px (RMSE) | "
                       f"Adjustments: {adjustment_key_presses} | Redraw: {redraw_time:.1f} ms")
        cv2.putText(frame, metric_text, (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        # Help panel
        if show_help:
            draw_help(frame, [
                "--- CONTROLS ---",
                "Translate (±2mm):  J/L (X), U/O (Y), I/K (Z)",
                "Rotate (±1°):      1/2 (Rx), 3/4 (Ry), 5/6 (Rz)",
                "Scale ×/÷ 1.01:    9 / 0",
                "Toggle fill/wire:  F",
                "Save T matrix:     S",
                "Save screenshot:   P",
                "Reset defaults:    R",
                "Quit:              Q / ESC",
                "",
                "(Auto-pick nearest marker; set TOP_ID >= 0 to force a specific ID.)"
            ], org=(10, 20))

        cv2.imshow(win, frame)
        k = cv2.waitKey(0) & 0xFF

        # Key handling (translation/rotation/scale/toggles).
        transform_key_pressed = False
        step_t = 0.002
        step_r = 1.0

        if   k == ord('j'): TX -= step_t; transform_key_pressed = True
        elif k == ord('l'): TX += step_t; transform_key_pressed = True
        elif k == ord('u'): TY -= step_t; transform_key_pressed = True
        elif k == ord('o'): TY += step_t; transform_key_pressed = True
        elif k == ord('i'): TZ += step_t; transform_key_pressed = True
        elif k == ord('k'): TZ -= step_t; transform_key_pressed = True
        elif k == ord('1'): RX -= step_r; transform_key_pressed = True
        elif k == ord('2'): RX += step_r; transform_key_pressed = True
        elif k == ord('3'): RY -= step_r; transform_key_pressed = True
        elif k == ord('4'): RY += step_r; transform_key_pressed = True
        elif k == ord('5'): RZ -= step_r; transform_key_pressed = True
        elif k == ord('6'): RZ += step_r; transform_key_pressed = True
        elif k == ord('9'):
            OBJ_SCALE_SKULL   *= 1.01
            OBJ_SCALE_LESION  *= 1.01
            OBJ_SCALE_MARKERS *= 1.01
        elif k == ord('0'):
            OBJ_SCALE_SKULL   /= 1.01
            OBJ_SCALE_LESION  /= 1.01
            OBJ_SCALE_MARKERS /= 1.01
        elif k == ord('f'):
            DRAW_FILLED = not DRAW_FILLED
        elif k == ord('r'):
            TX,TY,TZ = 0.000, 0.000, -0.060
            RX,RY,RZ = 90.0, 0.0, 0.0
            OBJ_SCALE_SKULL, OBJ_SCALE_LESION = 0.001, 0.001
            OBJ_SCALE_MARKERS = 0.001
            transform_key_pressed = True
        elif k == ord('s'):
            if not using_live_vars:
                print("Switching to live vars to save.")
                using_live_vars = True
                TX, TY, TZ = T_marker_skull_from_file[0:3, 3]
                print("WARNING: Saving with code-default RX,RY,RZ. Nudge rotation (e.g., 1 then 2) to persist file rotation.")
            R_align = euler_xyz_deg(RX, RY, RZ)
            T_marker_skull = make_T(R_align, [TX, TY, TZ])
            json.dump(T_marker_skull.tolist(), open(T_JSON_PATH, "w"), indent=2)
            print(f"[saved] {T_JSON_PATH}")
            print(f"--- Metrics: Alignment saved with {adjustment_key_presses} adjustments. ---")
            T_marker_skull_from_file = T_marker_skull
            using_live_vars = False
        elif k == ord('p'):
            out = "overlay_result.png"
            cv2.imwrite(out, frame)
            print(f"[saved] {out}")
        elif k == ord('h'):
            show_help = not show_help
        elif k in (ord('q'), 27):
            break

        # Transition to live-edit mode on first transform change.
        if transform_key_pressed:
            if not using_live_vars:
                print("Switched to live edit mode. Press 'S' to save changes.")
            using_live_vars = True
            adjustment_key_presses += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
