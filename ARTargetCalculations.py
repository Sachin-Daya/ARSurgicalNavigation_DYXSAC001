# ar_click_rmse.py (aka ARTargetCalculations.py)
# Workflow: click TARGET first, then ACTUAL for each landmark on a saved overlay image.
# Computes per-point pixel errors and summary stats (RMSE/mean/max), and writes:
#   • a CSV with all clicks and per-point errors
#   • an annotated PNG with vectors + summary text

import cv2, csv, os, math
import numpy as np
from pathlib import Path

# ------------- CONFIG -------------
IMAGE_PATH    = r"C:\Users\Sachin\OneDrive\Uni\4thYear\EEE4022S\Camera\ARResults\Intial13.png"   # saved overlay/screenshot to assess
OUT_DIR       = r"C:\Users\Sachin\OneDrive\Uni\4thYear\EEE4022S\Camera\AROutputsFinal"            # output folder for CSV + annotated image
OUT_BASENAME  = "manual_click_rmse"
N_POINTS_OPT  = None  # optional auto-stop after N pairs (e.g., 10); None = unlimited
POINT_RADIUS  = 5
COLOR_TARGET  = (0, 255, 255)  # yellow dot for TARGET (intended)
COLOR_ACTUAL  = (255, 0, 0)    # blue dot for ACTUAL (measured)
COLOR_VECTOR  = (60, 220, 60)  # green connector line TARGET→ACTUAL
FONT          = cv2.FONT_HERSHEY_SIMPLEX
# ----------------------------------

def draw_status(frame, pairs_done, rmse, mean_e, max_e, awaiting_target):
    """
    HUD overlay showing progress and current instruction (TARGET vs ACTUAL).
    Drawn with a black shadow for readability over bright imagery.
    """
    h, w = frame.shape[:2]
    lines = [
        f"Pairs: {pairs_done}",
        f"RMSE: {rmse:.2f}px  mean: {mean_e:.2f}px  max: {max_e:.2f}px" if pairs_done > 0 else "RMSE: --",
        f"Click: {'TARGET' if awaiting_target else 'ACTUAL'}",
        "[U] undo  [S] save  [Q] quit"
    ]
    y = 26
    for s in lines:
        cv2.putText(frame, s, (12, y), FONT, 0.7, (0,0,0), 3, cv2.LINE_AA)      # shadow
        cv2.putText(frame, s, (12, y), FONT, 0.7, (255,255,255), 1, cv2.LINE_AA) # text
        y += 28

def compute_errors(targets, actuals):
    """
    Returns:
      - list of per-point Euclidean errors [px]
      - RMSE over all pairs
      - mean error
      - max error
    """
    if len(targets)==0: return [], float("nan"), float("nan"), float("nan")
    diff = np.array(actuals, np.float32) - np.array(targets, np.float32)
    errs = np.sqrt((diff**2).sum(axis=1))
    rmse = float(np.sqrt((errs**2).mean()))
    mean_e = float(errs.mean())
    max_e  = float(errs.max())
    return errs.tolist(), rmse, mean_e, max_e

def main():
    # Load image to annotate/click on
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(IMAGE_PATH)
    base = img.copy()

    # Ensure output directory exists
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Window setup (resizable; start with half-res fallback)
    win = "Click RMSE — target then actual"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, max(960, img.shape[1]//2), max(540, img.shape[0]//2))

    # Use a local copy of the optional limit to avoid touching the global
    n_points_opt = N_POINTS_OPT

    # State buffers
    targets, actuals = [], []
    clicks = []              # raw sequence: TARGET, ACTUAL, TARGET, ACTUAL, ...
    awaiting_target = True   # toggles each click

    # Mouse callback: record clicks as float pixel coords
    def on_mouse(event, x, y, flags, userdata):
        nonlocal awaiting_target
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((float(x), float(y)))
            awaiting_target = not awaiting_target  # swap mode after each click

    cv2.setMouseCallback(win, on_mouse)

    while True:
        frame = base.copy()

        # Pairing logic: every two clicks form a (TARGET, ACTUAL) pair
        pairs = len(clicks)//2
        targets = clicks[0::2]
        actuals = clicks[1::2]

        # Draw points and vectors
        for pt in targets:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), POINT_RADIUS, COLOR_TARGET, -1, cv2.LINE_AA)
        for pt in actuals:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), POINT_RADIUS, COLOR_ACTUAL, -1, cv2.LINE_AA)
        for i in range(min(len(targets), len(actuals))):
            p = (int(targets[i][0]), int(targets[i][1]))
            q = (int(actuals[i][0]), int(actuals[i][1]))
            cv2.line(frame, p, q, COLOR_VECTOR, 2, cv2.LINE_AA)

        # Compute running stats on complete pairs only
        errs, rmse, mean_e, max_e = compute_errors(targets[:pairs], actuals[:pairs])
        draw_status(frame, pairs, rmse, mean_e, max_e, awaiting_target)

        cv2.imshow(win, frame)
        k = cv2.waitKey(15) & 0xFF

        # Key handling
        if k in (27, ord('q')):          # ESC or 'q' to exit
            break
        elif k == ord('u'):              # undo last click
            if clicks:
                clicks.pop()
                awaiting_target = not awaiting_target
        elif k == ord('s'):              # save CSV + annotated PNG
            if pairs == 0:
                print("Nothing to save yet.")
                continue
            csv_path = os.path.join(OUT_DIR, f"{OUT_BASENAME}_errors.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["index","target_u","target_v","actual_u","actual_v","error_px"])
                for i in range(pairs):
                    e = math.hypot(actuals[i][0]-targets[i][0], actuals[i][1]-targets[i][1])
                    w.writerow([i+1,
                                f"{targets[i][0]:.3f}", f"{targets[i][1]:.3f}",
                                f"{actuals[i][0]:.3f}", f"{actuals[i][1]:.3f}",
                                f"{e:.3f}"])
                w.writerow([])
                w.writerow(["RMSE_px", f"{rmse:.3f}"])
                w.writerow(["Mean_px", f"{mean_e:.3f}"])
                w.writerow(["Max_px",  f"{max_e:.3f}"])
            print(f"[csv] saved -> {csv_path}")

            out_img = os.path.join(OUT_DIR, f"{OUT_BASENAME}_overlay.png")
            annotated = frame.copy()
            txt = f"RMSE={rmse:.2f}px  mean={mean_e:.2f}px  max={max_e:.2f}px  (n={pairs})"
            cv2.putText(annotated, txt, (12, annotated.shape[0]-16), FONT, 0.8, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated, txt, (12, annotated.shape[0]-16), FONT, 0.8, (255,255,255), 1, cv2.LINE_AA)
            cv2.imwrite(out_img, annotated)
            print(f"[img] saved -> {out_img}")

        # Optional auto-stop when enough pairs are collected
        if n_points_opt is not None and pairs >= n_points_opt:
            print(f"Collected {pairs} pairs. Press S to save, or Q to quit.")
            # no mutation of n_points_opt or N_POINTS_OPT here

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
