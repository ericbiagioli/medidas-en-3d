import argparse
import json
import sys
from pathlib import Path
import tkinter as tk
import cv2
import numpy as np


root = tk.Tk()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()

def show_fitted(win, img):
    h, w = img.shape[:2]

    scale = min(screen_w / w, screen_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    cv2.imshow(win, resized)


def default_camera_matrix(w: int, h: int):
    focal_length_px = 0.8 * max(w, h)
    cx = w / 2.0
    cy = h/ 2.0
    K = np.array([[focal_length_px, 0, cx],
                  [0, focal_length_px, cy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist

def rvec_tvec_to_pose_dict(rvec, tvec):
    """
    Convert Rodrigues `rvec` and translation vector `tvec` into a json dict.
    Also, compute rotation matrix and yaw/pitch/roll (in degrees).
    """

    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    # convert to degrees
    euler_deg = {
        'roll_deg': float(np.degrees(x)),
        'pitch_deg': float(np.degrees(y)),
        'yaw_deg': float(np.degrees(z)),
    }

    return {
        'rvec': rvec.reshape(-1).tolist(),
        'tvec': tvec.reshape(-1).tolist(),
        'rotation_matrix': R.reshape(3, 3).tolist(),
        'euler_deg': euler_deg,
    }


def estimate_aruco_poses(image,
  marker_length_m: float,
  camera_matrix,
  dist_coeffs,
  marker_id: int | None = None):

    """Detect ArUco markers and estimate pose for each detected marker.

    Parameters
    - image: BGR image (numpy array)
    - marker_length_m: marker side length in meters
    - camera_matrix: 3x3 numpy array
    - dist_coeffs: distortion coefficients (array)
    - marker_id: if provided, only return poses for this id

    Returns a list of dicts: [{'id': int, 'corners': [...], 'pose': {...}}, ...]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

    params = cv2.aruco.DetectorParameters()

    # detect markers
    corners_list, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    results = []
    if ids is None or len(ids) == 0:
        return results

    # estimate pose for each marker
    # note: estimatePoseSingleMarkers returns rvecs (N,1,3) and tvecs (N,1,3)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners_list, marker_length_m, camera_matrix, dist_coeffs)

    for i, marker_id_found in enumerate(ids.flatten()):
        if (marker_id is not None) and (marker_id_found != marker_id):
            continue
        rvec = rvecs[i].reshape(3, 1)
        tvec = tvecs[i].reshape(3, 1)
        corners = corners_list[i].reshape(-1, 2).tolist()
        pose = rvec_tvec_to_pose_dict(rvec, tvec)
        results.append({'id': int(marker_id_found), 'corners': corners, 'pose': pose})

    return results


# ------------------------ CLI / Demo ------------------------

def draw_results(image, results, camera_matrix, dist_coeffs, marker_length_m):
    out = image.copy()
    for r in results:
        id_ = r['id']
        # draw marker corners
        corners = np.array(r['corners'], dtype=np.float32).reshape(-1, 2)
        corners_i = corners.astype(int)
        cv2.polylines(out, [corners_i], True, (0, 255, 0), 2)
        # draw id
        c0 = corners_i[0]
        cv2.putText(out, f"ID {id_}", (c0[0], c0[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # draw axis using the pose
        rvec = np.array(r['pose']['rvec'], dtype=np.float64).reshape(3, 1)
        tvec = np.array(r['pose']['tvec'], dtype=np.float64).reshape(3, 1)
        #cv2.aruco.drawAxis(out, camera_matrix, dist_coeffs, rvec, tvec, marker_length_m * 0.5)
        cv2.drawFrameAxes(out, camera_matrix, dist_coeffs, rvec, tvec, marker_length_m * 0.5)

    return out


def parse_args():
    p = argparse.ArgumentParser(description='Estimate ArUco pose for known-size marker')
    p.add_argument('--image', help='Path to input image. If omitted, webcam is used')
    p.add_argument('--marker-size', type=float, default=0.045, required=True, help='Marker side length in meters (e.g. 0.045)')
    p.add_argument('--marker-id', type=int, default=63, help='If provided, only look for this marker ID')
    return p.parse_args()


def main():
    args = parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Image {img_path} doesn't exist", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(str(img_path))

    h, w = image.shape[:2]

    camera_matrix, dist_coeffs = default_camera_matrix(w, h)

    results = estimate_aruco_poses(image=image,
      marker_length_m = 0.045,
      camera_matrix=camera_matrix,
      dist_coeffs=dist_coeffs,
      marker_id=63
    )

    print("results = ")
    print(json.dumps(results, indent=2))

    # draw and show
    out = draw_results(image, results, camera_matrix, dist_coeffs, args.marker_size)

    winname = "aruco result"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    show_fitted(winname, out)
    print('Press any key in the image window to exit')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()




winname="blahblah"
device='/dev/video0'

cap = cv2.VideoCapture(device)

if not cap.isOpened():
    print("No se pudo abrir la cámara /dev/video2")
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer el frame de la cámara")
    else:
        cv2.imshow(winname, frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

