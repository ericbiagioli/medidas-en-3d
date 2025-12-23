import cv2
import numpy as np
import tkinter as tk
from types import SimpleNamespace

from helpers import show_fitted
from helpers import get_distance
from helpers import list_cameras

# Reemplazá estas matrices por las TUYAS
# P1 y P2 son matrices de proyección 3x4
####P1 = np.array([
####    [700, 0, 640,   0],
####    [0, 700, 360,   0],
####    [0,   0,   1,   0]
####], dtype=np.float64)
####
####P2 = np.array([
####    [700, 0, 640, -280],   # baseline ~0.40m * focal
####    [0, 700, 360,    0],
####    [0,   0,   1,    0]
####], dtype=np.float64)
####
data = np.load("stereo_charuco_calibration.npz")

K1 = data["K1"]
D1 = data["D1"]
K2 = data["K2"]
D2 = data["D2"]
P1 = data["P1"]
P2 = data["P2"]


MARKER_SIZE = 0.043

def process_video(params):
  capL = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)
  capR = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)

  L_w = capL.get(cv2.CAP_PROP_FRAME_WIDTH)
  L_h = capL.get(cv2.CAP_PROP_FRAME_HEIGHT)
  #print("L_w = ", L_w, "   L_h = ", L_h)

  while True:
    ret_L, frame_L = capL.read()
    ret_R, frame_R = capR.read()
    if not ret_L:
      print("No se pudo leer el frame de la cámara L")
    if not ret_R:
      print("No se pudo leer el frame de la cámara R")
    if not ret_L or not ret_R:
      continue

    show_fitted(params.winname_L, frame_L)
    show_fitted(params.winname_R, frame_R)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
      break

  cap.release()

#def triangulate_points(pt_l, pt_r):
#    """
#    Triangula un punto 3D a partir de dos puntos 2D
#    """
#    pt_l = pt_l.reshape(2, 1)
#    pt_r = pt_r.reshape(2, 1)
#
#    X = cv2.triangulatePoints(P1, P2, pt_l, pt_r)
#    X = X[:3] / X[3]
#
#    return X.flatten()

def triangulate_points(pts_l, pts_r):
    """
    pts_l, pts_r: (4,2)
    devuelve: (4,3)
    """
    pts_3d = []

    for i in range(4):
        pl = pts_l[i].reshape(2, 1)
        pr = pts_r[i].reshape(2, 1)

        X = cv2.triangulatePoints(P1, P2, pl, pr)
        X = (X[:3] / X[3]).flatten()
        pts_3d.append(X)

    return np.array(pts_3d)



def detect_marker_center(frame):
    """
    Detecta el primer ArUco y devuelve:
    - centro (x,y) subpíxel
    - id del marcador
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, None

    # Tomamos el primer marcador
    pts = corners[0][0]  # (4,2)
    center = pts.mean(axis=0)

    return center.astype(np.float32), ids[0][0]

def detect_marker(frame):
    """
    Devuelve:
    - corners (4,2)
    - id
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, None

    return corners[0][0].astype(np.float32), ids[0][0]



def estimate_pose_from_3d(pts_3d):
    """
    Ajusta R,t a partir de puntos 3D
    """

    s = MARKER_SIZE / 2.0

    # Sistema de coordenadas del marcador
    obj_pts = np.array([
        [-s,  s, 0],
        [ s,  s, 0],
        [ s, -s, 0],
        [-s, -s, 0]
    ], dtype=np.float64)

    # Centrar ambos conjuntos
    obj_centroid = obj_pts.mean(axis=0)
    cam_centroid = pts_3d.mean(axis=0)

    obj_pts_c = obj_pts - obj_centroid
    cam_pts_c = pts_3d - cam_centroid

    # SVD (Kabsch)
    H = obj_pts_c.T @ cam_pts_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cam_centroid - R @ obj_centroid

    return R, t


def draw_axes(frame, corners):
    c = corners.mean(axis=0).astype(int)
    cv2.line(frame, c, c + (30, 0), (0,0,255), 2)
    cv2.line(frame, c, c + (0,30), (0,255,0), 2)


ar_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
ar_pars = cv2.aruco.DetectorParameters()

ar_pars.adaptiveThreshWinSizeMin = 5
ar_pars.adaptiveThreshWinSizeMax = 23
ar_pars.adaptiveThreshWinSizeStep = 4

ar_pars.adaptiveThreshConstant = 7

ar_pars.minCornerDistanceRate = 0.05
ar_pars.minMarkerPerimeterRate = 0.03
ar_pars.maxMarkerPerimeterRate = 4.0

ar_pars.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
ar_pars.cornerRefinementWinSize = 7
ar_pars.cornerRefinementMaxIterations = 100
ar_pars.cornerRefinementMinAccuracy = 1e-6

detector = cv2.aruco.ArucoDetector(ar_dict, ar_pars)

def main2():
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    params = SimpleNamespace(
        winname_L="Webcam_L",
        winname_R="Webcam_R",
        screen_w=screen_w,
        screen_h=screen_h
    )

    half_w = screen_w // 2
    full_h = screen_h

    # Crear ventanas
    cv2.namedWindow(params.winname_L, cv2.WINDOW_NORMAL)
    cv2.namedWindow(params.winname_R, cv2.WINDOW_NORMAL)

    # Redimensionar ventanas
    cv2.resizeWindow(params.winname_L, half_w, full_h)
    cv2.resizeWindow(params.winname_R, half_w, full_h)

    # Posicionar ventanas (horizontal)
    cv2.moveWindow(params.winname_L, 0, 0)
    cv2.moveWindow(params.winname_R, half_w, 0)

    CAM_LEFT = 2
    CAM_RIGHT = 4

    cap_l = cv2.VideoCapture(f"/dev/video{CAM_LEFT}")
    cap_r = cv2.VideoCapture(f"/dev/video{CAM_RIGHT}")

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("❌ No se pudieron abrir las cámaras")
        return

    print("▶ Presioná ESC para salir")

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l or not ret_r:
            print("❌ Error leyendo cámaras")
            break

#        center_l, id_l = detect_marker_center(frame_l)
#        center_r, id_r = detect_marker_center(frame_r)
#
#        if center_l is not None and center_r is not None:
#            if id_l == id_r:
#                X = triangulate_point(center_l, center_r)
#
#                x, y, z = X
#                print(f"ID {id_l} → X={x:.4f}  Y={y:.4f}  Z={z:.4f} m")
#
#                # Dibujar centro
#                cv2.circle(frame_l, tuple(center_l.astype(int)), 5, (0,255,0), -1)
#                cv2.circle(frame_r, tuple(center_r.astype(int)), 5, (0,255,0), -1)

        pts_l, id_l = detect_marker(frame_l)
        pts_r, id_r = detect_marker(frame_r)

        if pts_l is not None and pts_r is not None and id_l == id_r:

            pts_3d = triangulate_points(pts_l, pts_r)
            R, t = estimate_pose_from_3d(pts_3d)

            #print(f"\nID {id_l}")
            #print("R =")
            #print(R)
            #print("t =", t)

            draw_axes(frame_l, pts_l)
            draw_axes(frame_r, pts_r)

        show_fitted(params.winname_L, frame_l)
        show_fitted(params.winname_R, frame_r)

        if cv2.waitKey(1) == 27:
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()



def main():
    # Obtener tamaño de pantalla usando Tk
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    params = SimpleNamespace(
        winname_L="Webcam_L",
        winname_R="Webcam_R",
        screen_w=screen_w,
        screen_h=screen_h
    )

    half_w = screen_w // 2
    full_h = screen_h

    # Crear ventanas
    cv2.namedWindow(params.winname_L, cv2.WINDOW_NORMAL)
    cv2.namedWindow(params.winname_R, cv2.WINDOW_NORMAL)

    # Redimensionar ventanas
    cv2.resizeWindow(params.winname_L, half_w, full_h)
    cv2.resizeWindow(params.winname_R, half_w, full_h)

    # Posicionar ventanas (horizontal)
    cv2.moveWindow(params.winname_L, 0, 0)
    cv2.moveWindow(params.winname_R, half_w, 0)

    # Procesamiento principal
    process_video(params)

    cv2.destroyAllWindows()


#print(list_cameras())
#exit(0)

print("P1 = ", P1)
print("P2 = ", P2)

main2()

