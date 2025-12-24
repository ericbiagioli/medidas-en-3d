import cv2
import numpy as np
import glob
import os

import time
# ============================================================
# CONFIGURACIÓN
# ============================================================

# ChArUco (AJUSTAR A TU TABLERO REAL)
CHARUCO_ROWS = 9
CHARUCO_COLS = 6
SQUARE_LENGTH = 0.030  # metros
MARKER_LENGTH = 0.022  # metros
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

NUM_FRAMES = 50        # mínimo razonable: 20–30
SAVE_DIR = "stereo_calibration_charuco_frames"

# ============================================================
# ARUCO / CHARUCO
# ============================================================

aruco_dict = cv2.aruco.getPredefinedDictionary( cv2.aruco.DICT_4X4_50)

board = cv2.aruco.CharucoBoard(
    (CHARUCO_COLS, CHARUCO_ROWS),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)

params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

# ============================================================
# UTILIDADES
# ============================================================

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def detect_charuco(frame, debug=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, None

    _, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )

    if char_ids is None or len(char_ids) < 6:
        return None, None

    if debug:
        cv2.aruco.drawDetectedCornersCharuco(frame, char_corners, char_ids)
        cv2.imshow("Charuco Debug", frame)
        cv2.waitKey(500)

    return char_corners, char_ids

# ============================================================
# CAPTURA DE FRAMES
# ============================================================

def capture_frames(SAVE_DIR="stereo_calibration_charuco_frames",
  CAM_LEFT="/dev/video0",
  CAM_RIGHT="/dev/video2",
  MIN_COMMON_IDS=12,
  MIN_MOVE_PX=15,
  COOLDOWN=0.5, resolution=(640, 480)):

    mkdir(SAVE_DIR)
    cap_l = cv2.VideoCapture(CAM_LEFT)
    cap_r = cv2.VideoCapture(CAM_RIGHT)
    # Mitigar el delay que hay entre frames.
    cap_l.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_r.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_l.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap_l.set(cv2.CAP_PROP_FPS, 30)
    cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap_r.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap_r.set(cv2.CAP_PROP_FPS, 30)
    cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    valid_left   = 0
    valid_right  = 0
    valid_stereo = 0
    saved        = 0

    last_save_time = 0
    last_centroid = None

    print("▶ Captura automática activada (ESC para salir)\n")

    while saved < NUM_FRAMES:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        cl, il = detect_charuco(frame_l)
        cr, ir = detect_charuco(frame_r)

        ok_l = cl is not None
        ok_r = cr is not None

        ok_s = False
        n_common = 0

        if ok_l and ok_r:
            common_ids = np.intersect1d(il.flatten(), ir.flatten())
            n_common = len(common_ids)
            ok_s = n_common >= MIN_COMMON_IDS

        # --- Criterio de movimiento (vista distinta) ---
        centroid = None
        moved = True

        if ok_l:
            centroid = np.mean(cl, axis=0).ravel()

            if last_centroid is not None:
                moved = np.linalg.norm(centroid - last_centroid) > MIN_MOVE_PX

        # --- Criterio temporal ---
        now = time.time()
        cooldown_ok = (now - last_save_time) > COOLDOWN

        auto_capture = ok_l and ok_r and ok_s and moved and cooldown_ok

        # ---- DIBUJO ----
        vis_l = frame_l.copy()
        vis_r = frame_r.copy()

        if ok_l:
            cv2.aruco.drawDetectedCornersCharuco(vis_l, cl, il)
        if ok_r:
            cv2.aruco.drawDetectedCornersCharuco(vis_r, cr, ir)

        color = (0,255,0) if auto_capture else (0,0,255)

        cv2.putText(
            vis_l,
            f"L:{valid_left} R:{valid_right} S:{valid_stereo}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            vis_l,
            f"Common IDs: {n_common}",
            (10,65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.imshow(CAM_LEFT, vis_l)
        cv2.imshow(CAM_RIGHT, vis_r)

        # ---- AUTO SAVE ----
        if auto_capture:
            cv2.imwrite(f"{SAVE_DIR}/left_{saved:02d}.png", frame_l)
            cv2.imwrite(f"{SAVE_DIR}/right_{saved:02d}.png", frame_r)

            valid_left   += 1
            valid_right  += 1
            valid_stereo += 1
            saved += 1

            last_save_time = now
            last_centroid = centroid

            print(
                f"✔ Auto-captura {saved:02d} | "
                f"IDs comunes: {n_common}"
            )

        if cv2.waitKey(1) == 27:
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()





# ============================================================
# CALIBRACIÓN MONOCULAR
# ============================================================

def calibrate_single(image_files):
    all_corners = []
    all_ids = []
    image_size = None

    for fname in image_files:
        img = cv2.imread(fname)
        if image_size is None:
            image_size = img.shape[1], img.shape[0]

        corners, ids = detect_charuco(img,debug=True)
        if corners is not None:
            all_corners.append(corners)
            all_ids.append(ids)

    # --- Inicialización segura de K ---
    fx = fy = 0.8 * image_size[0]
    cx = image_size[0] / 2
    cy = image_size[1] / 2

    K_init = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,   0,  1]
    ], dtype=np.float64)

    D_init = np.zeros((5, 1))


    ret, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners,
        all_ids,
        board,
        image_size,
        K_init,
        D_init,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )

    print("✔ RMS:", ret)

    if len(all_corners) == 0:
        raise RuntimeError(
            "❌ No se detectaron esquinas ChArUco en ninguna imagen.\n"
            "   Revisá iluminación, tamaño del tablero y parámetros."
        )

    return K, D, image_size


def stereo_calibrate(K1, D1, K2, D2, image_size):

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    left_images  = sorted(glob.glob(f"{SAVE_DIR}/left_*.png"))
    right_images = sorted(glob.glob(f"{SAVE_DIR}/right_*.png"))

    for lf, rf in zip(left_images, right_images):
        img_l = cv2.imread(lf)
        img_r = cv2.imread(rf)

        cl, il = detect_charuco(img_l)
        cr, ir = detect_charuco(img_r)

        if cl is None or cr is None:
            print("⚠️ ChArUco no detectado en ambos")
            continue

        common_ids = np.intersect1d(il.flatten(), ir.flatten())
        print("IDs comunes:", common_ids)

        if len(common_ids) < 6:
            print("⚠️ Muy pocos puntos comunes")
            continue

        obj = []
        pts_l = []
        pts_r = []

        charuco_3d = board.getChessboardCorners()

        for cid in common_ids:
            idx_l = np.where(il == cid)[0][0]
            idx_r = np.where(ir == cid)[0][0]

            #obj.append(board.chessboardCorners[cid])
            obj.append(charuco_3d[cid])
            pts_l.append(cl[idx_l])
            pts_r.append(cr[idx_r])

        objpoints.append(np.array(obj, dtype=np.float32))
        imgpoints_l.append(np.array(pts_l, dtype=np.float32))
        imgpoints_r.append(np.array(pts_r, dtype=np.float32))

    print(f"\n✔ Vistas estéreo válidas: {len(objpoints)}")

    if len(objpoints) == 0:
        raise RuntimeError("❌ No hay vistas estéreo válidas")

    flags = cv2.CALIB_FIX_INTRINSIC

    ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        K1, D1,
        K2, D2,
        image_size,
        flags=flags
    )

    print("✔ Stereo RMS:", ret)
    return R, T



# ============================================================
# MAIN
# ============================================================

def main():

    capture_frames(SAVE_DIR="stereo_calibration_charuco_frames",
      CAM_LEFT="/dev/video0",
      CAM_RIGHT="/dev/video2",
      MIN_COMMON_IDS=12,
      MIN_MOVE_PX=15,
      COOLDOWN=2.0)

    left_images  = sorted(glob.glob(f"{SAVE_DIR}/left_*.png"))
    right_images = sorted(glob.glob(f"{SAVE_DIR}/right_*.png"))

    print("\n▶ Calibrando cámara izquierda")
    K1, D1, image_size = calibrate_single(left_images)

    print("\n▶ Calibrando cámara derecha")
    K2, D2, _ = calibrate_single(right_images)

    print("\n▶ Calibración estéreo")
    R, T = stereo_calibrate(K1, D1, K2, D2, image_size)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1,
        K2, D2,
        image_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY
    )

    np.savez(
        "stereo_charuco_calibration.npz",
        K1=K1, D1=D1,
        K2=K2, D2=D2,
        R=R, T=T,
        R1=R1, R2=R2,
        P1=P1, P2=P2,
        Q=Q
    )

    print("\n🎉 CALIBRACIÓN COMPLETA")
    print("P1 =\n", P1)
    print("P2 =\n", P2)

if __name__ == "__main__":
    main()

