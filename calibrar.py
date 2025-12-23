import cv2
import numpy as np
import pickle

# =========================
# CONFIGURACIÓN DEL TABLERO
# =========================

CHARUCO_ROWS = 8
CHARUCO_COLS = 11
CHECKER_SIZE = 0.015   # metros (15 mm)
MARKER_SIZE  = 0.011   # metros (11 mm)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

board = cv2.aruco.CharucoBoard(
    (CHARUCO_COLS, CHARUCO_ROWS),
    CHECKER_SIZE,
    MARKER_SIZE,
    ARUCO_DICT
)

detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, detector_params)

# =========================
# CAPTURA DE DATOS
# =========================

cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

all_charuco_corners = []
all_charuco_ids = []

print("\nPresiona 'c' para capturar una vista válida")
print("Presiona 'q' para calibrar y salir\n")

winname="calibracion"
cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

image_seq = 0
capturing = False
while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame de la cámara")
        exit(0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco_detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco( corners, ids, gray, board)

        if retval > 20:
            cv2.aruco.drawDetectedCornersCharuco( frame, charuco_corners, charuco_ids)

    cv2.imshow(winname, frame)

    if capturing:
      if ids is not None and retval > 20:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        cv2.imwrite(f"imagen-{image_seq}.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        image_seq = image_seq + 1
        capturing = False
        print(f"✔ Captura guardada ({len(all_charuco_corners)})")

    k = cv2.waitKey(1) & 0xFF

    if k == ord('c'):
      capturing = True
    elif k == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()

# =========================
# CALIBRACIÓN
# =========================

if len(all_charuco_corners) < 5:
    raise RuntimeError("No hay suficientes capturas para calibrar")

image_size = gray.shape[::-1]

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

# =========================
# RESULTADOS
# =========================

print("\n📷 RESULTADOS DE CALIBRACIÓN")
print("----------------------------")
print("RMS reprojection error:", ret)
print("\nCamera matrix:\n", camera_matrix)
print("\nDistortion coefficients:\n", dist_coeffs)

with open("calibration_charuco.pkl", "wb") as f:
    pickle.dump((camera_matrix, dist_coeffs), f)

print("\n✔ Calibración guardada en calibration_charuco.pkl")

