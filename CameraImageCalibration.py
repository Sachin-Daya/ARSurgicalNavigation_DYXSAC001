import cv2
import numpy as np
import glob
import os
from pathlib import Path

# --- INPUTS (edit if needed) ---
# Folder pattern for calibration images. Keep the trailing wildcard for file type.
IMAGE_GLOB = r"C:\Users\Sachin\OneDrive\Uni\4thYear\EEE4022S\Camera\CalibrationImages15\*.jpg"

# Inner-corner grid size as (columns, rows). Values refer to the number of corner intersections.
# Common checkerboard used in practice is 9x6 inner corners.
CHESSBOARD = (9, 6)

# Physical edge length of one checkerboard square in meters (e.g., 28 mm -> 0.028 m).
SQUARE_SIZE_M = 0.028

# Output artifact paths
OUT_NPZ = "calibration_data_ipad.npz"
OUT_YAML = "calibration_data_ipad.yml"
# -------------------------------

# Prepare the 3D object-point template for one checkerboard view, scaled by square size.
# Coordinates lie on the z=0 plane with (x, y) spanning the grid.
objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_M

# Accumulators for 3D world points and 2D detected image points across all images.
objpoints = []  # 3D points in world coordinates
imgpoints = []  # 2D points in pixel coordinates

# Sub-pixel corner refinement criteria and initial corner-detection flags.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
         cv2.CALIB_CB_NORMALIZE_IMAGE |
         cv2.CALIB_CB_FAST_CHECK)

# Collect candidate image paths and ensure the set is non-empty.
image_paths = sorted(glob.glob(IMAGE_GLOB))
if not image_paths:
    raise FileNotFoundError(f"No images matched: {IMAGE_GLOB}")

# Iterate through images, detect checkerboard corners, and refine to sub-pixel accuracy.
h, w = None, None
kept = 0
for p in image_paths:
    img = cv2.imread(p)
    if img is None:
        print(f"[skip] Could not read {p}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if h is None:
        h, w = gray.shape[:2]
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, flags)
    if not ret:
        print(f"[no corners] {Path(p).name}")
        continue

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners2)
    kept += 1

    # Optional visual QA: overlay detected corners and save alongside the original for inspection.
    vis = img.copy()
    cv2.drawChessboardCorners(vis, CHESSBOARD, corners2, True)
    cv2.imwrite(os.path.join(Path(p).parent, f"_corners_{Path(p).name}"), vis)

print(f"\nDetected corners in {kept}/{len(image_paths)} images.")
# Zhang-style calibration typically benefits from ≥10 views with varied poses.
if kept < 10:
    print("[warn] Fewer than 10 good detections — calibration may be unstable.")

# Run intrinsic calibration to estimate camera matrix (K) and distortion coefficients.
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (w, h), None, None
)

# Compute overall RMSE reprojection error across all views for a quick quality check.
tot_err = 0
tot_pts = 0
for i in range(len(objpoints)):
    imgpts2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    err = cv2.norm(imgpoints[i], imgpts2, cv2.NORM_L2)
    tot_err += err*err
    tot_pts += len(objpoints[i])
rmse = np.sqrt(tot_err / tot_pts)

print("\n=== Calibration Results ===")
print("Image size:", (w, h))
print("RMS reprojection error (px):", rmse)
print("Camera matrix K:\n", K)
print("Distortion coeffs [k1 k2 p1 p2 k3 ...]:\n", dist.ravel())

# Persist results in NPZ for downstream Python workflows.
np.savez(OUT_NPZ, K=K, dist=dist, rvecs=rvecs, tvecs=tvecs, image_size=(w, h))
print(f"\nSaved NPZ -> {OUT_NPZ}")

# Also export an OpenCV YAML for compatibility with non-Python tools.
fs = cv2.FileStorage(OUT_YAML, cv2.FILE_STORAGE_WRITE)
fs.write("image_width", int(w))
fs.write("image_height", int(h))
fs.write("camera_matrix", K)
fs.write("distortion_coefficients", dist)
fs.release()
print(f"Saved YAML -> {OUT_YAML}")

# Create a side-by-side undistortion preview using the first image with a valid detection.
first_ok = None
for p in image_paths:
    if Path(p).name.startswith("_corners_"):
        continue
    img = cv2.imread(p)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, _ = cv2.findChessboardCorners(gray, CHESSBOARD, flags)
    if ret2:
        first_ok = img
        break

if first_ok is not None:
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    und = cv2.undistort(first_ok, K, dist, None, newK)
    side_by_side = np.hstack([first_ok, und])
    out_preview = "undistort_preview.png"
    cv2.imwrite(out_preview, side_by_side)
    print(f"Undistort preview saved -> {out_preview} (left=original, right=undistorted)")
else:
    print("No preview saved (no successful image found for display).")
