import cv2
import numpy as np
import tkinter as tk
from types import SimpleNamespace

from helpers import show_fitted
from helpers import get_distance
from helpers import list_cameras

data = np.load("stereo_charuco_calibration.npz")

K1 = data["K1"]
D1 = data["D1"]
K2 = data["K2"]
D2 = data["D2"]
R = data["R"]
T = data["T"]
R1 = data["R1"]
R2 = data["R2"]
P1 = data["P1"]
P2 = data["P2"]
Q = data["Q"]

MARKER_SIZE = 0.043


# def project_with_P(P, X):
#    X = X.flatten()           # 🔑 CLAVE
#    Xh = np.append(X, 1.0)    # (4,)
#    x = P @ Xh
#    return (x[:2] / x[2]).astype(np.float32)


def project_with_P(P, X):
    Xh = np.append(X, 1.0)  # X es (3,)
    x = P @ Xh  # OK: (3x4)@(4,)
    return (x[:2] / x[2]).astype(np.float32)


# def project_with_P(P, X):
#    Xh = np.hstack([X, 1.0])      # (4,)
#    x = P @ Xh                    # (3,)
#    return (x[:2] / x[2]).astype(np.float32)


def reproject_left(X):
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))

    pt, _ = cv2.projectPoints(X.reshape(1, 1, 3), rvec, tvec, K1, D1)
    return pt.reshape(2)


def reproject_right(X):
    rvec, _ = cv2.Rodrigues(R)
    tvec = T.reshape(3, 1)

    pt, _ = cv2.projectPoints(X.reshape(1, 1, 3), rvec, tvec, K2, D2)
    return pt.reshape(2)


def reproject_points(points_3d, K, D, R, t):
    """
    points_3d: (N,3)
    devuelve: (N,2)
    """
    rvec, _ = cv2.Rodrigues(R)
    imgpts, _ = cv2.projectPoints(
        points_3d.astype(np.float64), rvec, t.reshape(3, 1), K, D
    )
    return imgpts.reshape(-1, 2)


def triangulate_point(pt_l, pt_r):
    """
    pt_l, pt_r: (2,)
    devuelve: (3,)
    """
    pl = pt_l.reshape(2, 1)
    pr = pt_r.reshape(2, 1)

    X = cv2.triangulatePoints(P1, P2, pl, pr)
    X = (X[:3] / X[3]).flatten()
    return X


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


def detect_marker_center_id(frame):
    """
    Detecta el primer ArUco y devuelve:
    - corners (4,2)
    - centro (x,y) subpíxel
    - id del marcador
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, None, None

    # Tomamos el primer marcador
    pts = corners[0][0]  # (4,2)
    center = pts.mean(axis=0)

    return corners[0][0].astype(np.float32), center.astype(np.float32), ids[0][0]


def estimate_pose_from_3d(pts_3d):
    """
    Ajusta R,t a partir de puntos 3D
    """

    s = MARKER_SIZE / 2.0

    # Sistema de coordenadas del marcador
    obj_pts = np.array(
        [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64
    )

    # Centrar ambos conjuntos
    obj_centroid = obj_pts.mean(axis=0)
    cam_centroid = pts_3d.mean(axis=0)

    obj_pts_c = obj_pts - obj_centroid
    cam_pts_c = pts_3d - cam_centroid

    # SVD (Kabsch)
    H = obj_pts_c.T @ cam_pts_c
    U, _, Vt = np.linalg.svd(H)
    rvec = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        rvec = Vt.T @ U.T

    t = cam_centroid - rvec @ obj_centroid

    return rvec, t


def draw_axes(frame, corners):
    c = corners.mean(axis=0).astype(int)
    cv2.line(frame, c, c + (30, 0), (0, 0, 255), 2)
    cv2.line(frame, c, c + (0, 30), (0, 255, 0), 2)


def main(resolution, debug=True):
    print("K1\n", K1)
    print("P1\n", P1)
    print("image size assumed by P1 =", P1[0, 2] * 2, P1[1, 2] * 2)
    print("image size assumed by P2 =", P2[0, 2] * 2, P2[1, 2] * 2)

    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    params = SimpleNamespace(
        winname_L="Webcam_L",
        winname_R="Webcam_R",
        screen_w=screen_w,
        screen_h=screen_h,
        cam_left="/dev/video2",
        cam_right="/dev/video0",
    )

    half_w = screen_w // 2
    full_h = screen_h

    # Create windows
    cv2.namedWindow(params.winname_L, cv2.WINDOW_NORMAL)
    cv2.namedWindow(params.winname_R, cv2.WINDOW_NORMAL)

    # Resize windows
    cv2.resizeWindow(params.winname_L, half_w, full_h)
    cv2.resizeWindow(params.winname_R, half_w, full_h)

    # Positionate windows (tile horizontal)
    cv2.moveWindow(params.winname_L, 0, 0)
    cv2.moveWindow(params.winname_R, half_w, 0)

    # Open VideoCaptures
    cap_l = cv2.VideoCapture(params.cam_left)
    cap_r = cv2.VideoCapture(params.cam_right)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("❌ No se pudieron abrir las cámaras")
        return

    # Mitigate delay between captures.
    # They won't be perfectly syncronized, but at least to the best try.
    # Set the buffering to 1 (the minimum, and discard 1 frame when reading).
    cap_l.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_r.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap_l.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap_l.set(cv2.CAP_PROP_FPS, 30)
    cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap_r.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap_r.set(cv2.CAP_PROP_FPS, 30)
    cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Show the resolution of each VideoCapture
    if debug:
        L_w = cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)
        L_h = cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT)
        R_w = cap_r.get(cv2.CAP_PROP_FRAME_WIDTH)
        R_h = cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Resolution Left Camera = {L_w}x{L_h}")
        print(f"Resolution Right Camera = {R_w}x{R_h}")

    print("▶ Presioná ESC para salir")

    h, w = resolution[1], resolution[0]

    map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)

    map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    while True:
        # Discard one frame (buffered)
        cap_l.grab()
        cap_r.grab()
        ret_l, _ = cap_l.retrieve()
        ret_r, _ = cap_r.retrieve()
        # Read the frame
        ret_l, frame_l = cap_l.read()
        ret_l, frame_r = cap_r.read()

        if not ret_l or not ret_r:
            if not ret_l:
                print("No se pudo leer el frame de la cámara L")
            if not ret_r:
                print("No se pudo leer el frame de la cámara R")
            continue

        frame_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        frame_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

        pts_l, center_l, id_l = detect_marker_center_id(frame_l)
        pts_r, center_r, id_r = detect_marker_center_id(frame_r)

        ## Draw the center
        if center_l is not None and center_r is not None and id_l == id_r:
            # 1) triangulación
            ##Xc = triangulate_point(center_l, center_r)

            # X = cv2.triangulatePoints(P1, P2, center_l, center_r)
            # pt_l = project_with_P(P1, X)
            # pt_r = project_with_P(P2, X)

            pl = center_l.reshape(2, 1)
            pr = center_r.reshape(2, 1)

            Xh = cv2.triangulatePoints(P1, P2, pl, pr)
            Xc = (Xh[:3] / Xh[3]).flatten()  # ahora sí (3,)
            pt_l = project_with_P(P1, Xc)
            pt_r = project_with_P(P2, Xc)

            # 2) reproyección
            # pt_l = reproject_left(Xc)
            # pt_r = reproject_right(Xc)
            ##pt_l = project_with_P(P1, Xc)
            ##pt_r = project_with_P(P2, Xc)

            # 3) dibujo
            cv2.circle(frame_l, tuple(pt_l.astype(int)), 5, (0, 255, 0), -1)
            cv2.circle(frame_r, tuple(pt_r.astype(int)), 5, (0, 255, 0), -1)

        # Draw the vertices
        # if pts_l is not None and pts_r is not None and len(pts_l) == 4 and len(pts_r) == 4:
        #  for i in range(4):
        #    #Xi = triangulate_point(pts_l[i], pts_r[i])
        #    Xi = cv2.triangulatePoints(P1, P2, pts_l[i], pts_r[i])

        #    pl = project_with_P(P1, Xi)
        #    pr = project_with_P(P2, Xi)

        #    cv2.circle(frame_l, tuple(pl.astype(int)), 4, (255,0,0), -1)
        #    cv2.circle(frame_r, tuple(pr.astype(int)), 4, (255,0,0), -1)

        ## Estimate tvec and rvec
        if pts_l is not None and pts_r is not None and id_l == id_r:
            pts_3d = triangulate_points(pts_l, pts_r)
            R, t = estimate_pose_from_3d(pts_3d)

            # print(f"\nID {id_l}")
            # print("R =")
            # print(R)
            # print("t =", t)

        ## Draw axis
        if pts_l is not None and pts_r is not None and id_l == id_r:
            draw_axes(
                frame_l, pts_l
            )  ## LO ESTA DIBUJANDO SIEMPRE HORIZONTAL Y VERTICAL
            draw_axes(
                frame_r, pts_r
            )  ## LO ESTA DIBUJANDO SIEMPRE HORIZONTAL Y VERTICAL

        show_fitted(params.winname_L, frame_l)
        show_fitted(params.winname_R, frame_r)

        if cv2.waitKey(1) == 27:
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


resolution = (640, 480)
main(resolution)
