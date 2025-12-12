import cv2
import numpy as np
from robolink import *   # RoboDK API
from robodk import *     # RoboDK matrix/pose tools
import matplotlib.pyplot as plt
from robolink import TargetReachError


# ---------- CONFIG ----------
# Where to save the webcam snapshot (same place you used your Paint image)
IMAGE_PATH = "../data/webcam_capture.png"

TABLE_FRAME_NAME = "table"         # name of frame in RoboDK
DRAW_POSITION_NAME = "DrawBase"    # target in RoboDK
ROBOT_NAME = "Motoman HP6"         # robot name in the tree

DRAW_AREA_MM = 220                 # size of drawing square on the table (mm)
Z_DRAW = 50                        # height above table to draw at (mm)
POINT_STEP = 5                     # take every Nth point to reduce density
CAM_INDEX = 0                      # webcam index for cv2.VideoCapture
# -----------------------------


def capture_from_webcam(out_path, cam_index=0):
    """Capture a single frame from webcam and save it to out_path."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with index {cam_index}")

    # Grab a couple of frames to let exposure/auto-focus settle
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Failed to read frame from webcam")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture final frame from webcam")

    # Optionally show the captured frame for debugging
    cv2.imshow("Captured", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(out_path, frame)
    print(f"[INFO] Saved webcam frame to: {out_path}")


def load_main_contour(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold: dark object on lighter background
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("No contours found in image")

    # Take the largest contour by area
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt.squeeze()

    # Subsample points to avoid crazy dense paths
    pts = pts[::POINT_STEP]

    return pts, img.shape[:2]  # pts, (h, w)


def pixel_to_table_mm(points, img_size, draw_area_mm):
    """
    Map pixel coordinates to table coordinates in mm.
    We linearly scale the contour bounding box to a
    centered square of size draw_area_mm x draw_area_mm.
    """
    h, w = img_size
    xs = points[:, 0].astype(np.float32)
    ys = points[:, 1].astype(np.float32)

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    # normalize to [0, 1]
    xs_norm = (xs - xmin) / (xmax - xmin + 1e-6)
    ys_norm = (ys - ymin) / (ymax - ymin + 1e-6)

    # scale to [-L/2, L/2]
    L = float(draw_area_mm)
    xs_mm = (xs_norm - 0.5) * L
    # invert Y so that image top corresponds to +Y on table
    ys_mm = (0.5 - ys_norm) * L

    return np.stack([xs_mm, ys_mm], axis=1)


def main():
    # ---- 0) Capture a fresh image from the webcam ----
    capture_from_webcam(IMAGE_PATH, CAM_INDEX)

    # ---- 1) Vision: image -> contour points in table coords ----
    pts_px, img_size = load_main_contour(IMAGE_PATH)
    pts_mm = pixel_to_table_mm(pts_px, img_size, DRAW_AREA_MM)

    # ---- 2) RoboDK connection ----
    RDK = Robolink()

    robot = RDK.Item(ROBOT_NAME, ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise Exception(f"Robot '{ROBOT_NAME}' not found in station")

    table_frame = RDK.Item(TABLE_FRAME_NAME, ITEM_TYPE_FRAME)
    if not table_frame.Valid():
        raise Exception(f"Frame '{TABLE_FRAME_NAME}' not found in station")

    draw_base_target = RDK.Item(DRAW_POSITION_NAME, ITEM_TYPE_TARGET)
    if not draw_base_target.Valid():
        raise Exception(
            f"Target '{DRAW_POSITION_NAME}' not found. Create it in RoboDK."
        )

    # Use table frame as reference
    robot.setPoseFrame(table_frame)

    base_pose = draw_base_target.Pose()

    followed_pts = []   # points the robot actually visited

    # First point
    x0, y0 = pts_mm[0]
    start_pose = base_pose * transl(float(x0), float(y0), 0.0)
    robot.MoveJ(start_pose)
    followed_pts.append((x0, y0))

    # Rest of contour
    for x_mm, y_mm in pts_mm[1:]:
        target_pose = base_pose * transl(float(x_mm), float(y_mm), 0.0)
        try:
            robot.MoveL(target_pose)
            followed_pts.append((x_mm, y_mm))
        except TargetReachError:
            print(f"Skipping unreachable point: ({x_mm:.1f}, {y_mm:.1f})")
            continue

    print("Finished drawing contour from webcam image.")

    # ---- 3) Plot intended vs followed paths ----
    pts_mm = np.array(pts_mm)
    followed_pts = np.array(followed_pts)

    plt.figure()
    plt.plot(pts_mm[:, 0], pts_mm[:, 1], label="Original contour", linewidth=1)
    plt.plot(followed_pts[:, 0], followed_pts[:, 1], "--", label="Robot path", linewidth=2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Contour vs robot path (webcam capture)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
