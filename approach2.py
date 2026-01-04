import cv2
import numpy as np
import tkinter as tk
from types import SimpleNamespace
import os
import glob

from helpers import *

# =========================
# Utilities
# =========================


def rigid_alignment(A, B):
    """
    A, B: (N,3)
    Encuentra R, t tal que R @ A + t ≈ B
    """
    A_mean = A.mean(axis=0)
    B_mean = B.mean(axis=0)

    A_c = A - A_mean
    B_c = B - B_mean

    H = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = B_mean - R @ A_mean
    return R, t


def project_rectified(P, X):
    """
    Proyección en imágenes rectificadas
    P: 3x4
    X: (3,)
    """
    Xh = np.hstack([X, 1.0])  # (4,)
    x = P @ Xh  # (3,)
    return (x[:2] / x[2]).astype(np.float32)


def triangulate_point_rectified(P1, P2, pt_l, pt_r):
    """
    pt_l, pt_r: (2,)
    return: (3,)
    """
    pl = pt_l.reshape(2, 1)
    pr = pt_r.reshape(2, 1)

    Xh = cv2.triangulatePoints(P1, P2, pl, pr)
    X = (Xh[:3] / Xh[3]).flatten()
    return X


# =========================
# ArUco
# =========================

ar_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
ar_pars = cv2.aruco.DetectorParameters()
ar_pars.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

detector = cv2.aruco.ArucoDetector(ar_dict, ar_pars)


def aruco_points_and_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None:
        return None, None

    # corners tiene shape (numero de ids, 1, 4, 2), donde la 1era dimensión
    # es el número de ids encontrados, la 2da es siempre "1", y existe
    # simplemente para compatibilidad con otras operaciones en OpenCV
    # la tercera corresponde a los vertices del aruco recorridos en sentido
    # horario, comenzando por el "arriba a la izquierda" (<arriba> del aruco)

    pts = corners[0][0].astype(np.float32)  # (4,2)
    # (topLeft, topRight, bottomRight, bottomLeft) = corners
    center = pts.mean(axis=0)

    return pts, center


def project_with_P(P, X):
    Xh = np.append(X, 1.0)  # X es (3,)
    x = P @ Xh  # OK: (3x4)@(4,)
    return (x[:2] / x[2]).astype(np.float32)


# =========================
# Main
# =========================


def rigid_alignment(A, B):
    """
    A, B: (N,3)
    Encuentra R, t tal que R @ A + t ≈ B
    """
    A_mean = A.mean(axis=0)
    B_mean = B.mean(axis=0)

    A_c = A - A_mean
    B_c = B - B_mean

    H = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = B_mean - R @ A_mean
    return R, t


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


##
## To do: tiene que pasarse como parámetro el aruco_3d, el aruco_sz y el needle_tip
##
def estimate_3d_position(
    fn_l, fn_r, calib, aruco_3d, needle_tip, visualization=True, img_size=(640, 480)
):
    # Calibration matrices
    data = np.load(calib)

    K1 = data["K1"]
    D1 = data["D1"]
    K2 = data["K2"]
    D2 = data["D2"]

    R1 = data["R1"]
    R2 = data["R2"]
    P1 = data["P1"]
    P2 = data["P2"]
    Q = data["Q"]

    # Images
    L = cv2.imread(fn_l)
    R = cv2.imread(fn_r)

    if L is None or R is None:
        raise Exception("No se pudo leer la imagen")

    # Rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

    L_rect = cv2.remap(L, map1x, map1y, cv2.INTER_LINEAR)
    R_rect = cv2.remap(R, map2x, map2y, cv2.INTER_LINEAR)

    ## Aruco points and centers, both in rectified and in original images
    points_L, center_float_L = aruco_points_and_center(L)
    points_R, center_float_R = aruco_points_and_center(R)
    points_L_rect, center_float_L_rect = aruco_points_and_center(L_rect)
    points_R_rect, center_float_R_rect = aruco_points_and_center(R_rect)

    if (
        points_L is None
        or points_R is None
        or points_L_rect is None
        or points_R_rect is None
    ):
        print("No se han detectado arucos")
        return None

    center_int_L = None
    center_int_R = None
    center_int_L_rect = None
    center_int_R_rect = None

    if center_float_L is not None:
        center_int_L = tuple(np.round(center_float_L).astype(int))
    if center_float_R is not None:
        center_int_R = tuple(np.round(center_float_R).astype(int))
    if center_float_L_rect is not None:
        center_int_L_rect = tuple(np.round(center_float_L_rect).astype(int))
    if center_float_R_rect is not None:
        center_int_R_rect = tuple(np.round(center_float_R_rect).astype(int))

    # 3D coordinates of the vertices of the aruco
    #
    # Produces 3D coordinates **IN THE COORDINATE SYSTEM IF THE CAMERA 1**
    # This is: the reference system of the left camera.
    #
    # El “mundo” coincide con:
    #
    # origen en el centro óptico de la cámara izquierda
    #
    # ejes alineados con la cámara izquierda
    #
    # +X → derecha en la imagen
    # +Y → abajo en la imagen
    # +Z → hacia adelante (frente a la cámara)
    #
    # Homogeneus coordinates
    # Xh = (x * w, y * w, z * w, w)

    points_3d_h = cv2.triangulatePoints(
        P1, P2, points_L_rect.T.astype(np.float64), points_R_rect.T.astype(np.float64)
    )
    # Euclidean coordinates
    # (x, y, z) = (Xh[0] / w, Xh[1] / w, Xh[2] / w)
    points_3d = (points_3d_h[:3] / points_3d_h[3]).T

    R_aruco, t_aruco = rigid_alignment(aruco_3d, points_3d)
    X_tip = R_aruco @ needle_tip + t_aruco

    ## Visualization
    if visualization:
        vis_L = L.copy()
        vis_R = R.copy()
        vis_L_rect = L_rect.copy()
        vis_R_rect = R_rect.copy()

        if center_int_L is not None:
            cv2.circle(vis_L, center_int_L, 8, (0, 255, 0), -1)
        if center_int_R is not None:
            cv2.circle(vis_R, center_int_R, 8, (0, 255, 0), -1)
        if center_int_L_rect is not None:
            cv2.circle(vis_L_rect, center_int_L_rect, 8, (0, 255, 0), -1)
        if center_int_R_rect is not None:
            cv2.circle(vis_R_rect, center_int_R_rect, 8, (0, 255, 0), -1)

        cv2.imshow("L", vis_L)
        cv2.imshow("R", vis_R)
        cv2.imshow("L_rect", vis_L_rect)
        cv2.imshow("R_rect", vis_R_rect)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return X_tip


def compute_measure_stereo(
    l1, r1, l2, r2, calib, aruco_3d, needle_tip, visualization=True
):
    p1 = estimate_3d_position(l1, r1, calib, aruco_3d, needle_tip, visualization)
    p2 = estimate_3d_position(l2, r2, calib, aruco_3d, needle_tip, visualization)
    return distance(p1, p2)


def files_ok(lefts, rights, pref_ls="left_", pref_rs="right_"):
    ls = [os.path.basename(l).replace(pref_ls, "") for l in lefts]
    rs = [os.path.basename(r).replace(pref_rs, "") for r in rights]
    return ls == rs


def test_all_pairs_same_set(lefts, rights, calib, expected, aruco_3d, needle_tip):
    assert files_ok(lefts, rights)

    pairs = list(zip(lefts, rights))

    for i, (l1, r1) in enumerate(pairs):
        for l2, r2 in pairs[i + 1 :]:
            detected = compute_measure_stereo(
                l1, r1, l2, r2, calib, aruco_3d, needle_tip
            )
            print(f"Expected: {expected}  Detected: {detected}")


def test_all_pairs_bipartite(
    lefts1, rights1, lefts2, rights2, calib, expected, aruco_3d, needle_tip
):
    assert files_ok(lefts1, rights1)
    assert files_ok(lefts2, rights2)

    for l1, r1 in zip(lefts1, rights1):
        for l2, r2 in zip(lefts2, rights2):
            detected = compute_measure_stereo(
                l1, r1, l2, r2, calib, aruco_3d, needle_tip
            )
            print(f"Expected: {expected}  Detected: {detected}")


def testset1():
    image_size = (640, 480)

    calib = "tests/testset1/stereo_charuco_calibration.npz"

    l_20 = sorted(glob.glob("tests/testset1/left_pos=20.0_*.png"))
    r_20 = sorted(glob.glob("tests/testset1/right_pos=20.0_*.png"))

    l_10 = sorted(glob.glob("tests/testset1/left_pos=10.0_*.png"))
    r_10 = sorted(glob.glob("tests/testset1/right_pos=10.0_*.png"))

    l_1 = sorted(glob.glob("tests/testset1/left_pos=1.0_*.png"))
    r_1 = sorted(glob.glob("tests/testset1/right_pos=1.0_*.png"))

    # Aruco and needle tip.
    aruco_sz = 0.043
    aruco_3d = np.array(
        [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=np.float32
    )
    aruco_3d = aruco_3d * (aruco_sz / 2.0)
    needle_tip = np.array([-(aruco_sz / 2.0) - 0.02, -0.145, -0.003])

    test_all_pairs_same_set(l_1, r_1, calib, 0.0, aruco_3d, needle_tip)
    test_all_pairs_same_set(l_10, r_10, calib, 0.0, aruco_3d, needle_tip)
    test_all_pairs_same_set(l_20, r_20, calib, 0.0, aruco_3d, needle_tip)
    test_all_pairs_bipartite(l_1, r_1, l_10, r_10, calib, 9.0, aruco_3d, needle_tip)
    test_all_pairs_bipartite(l_1, r_1, l_20, r_20, calib, 19.0, aruco_3d, needle_tip)
    test_all_pairs_bipartite(l_10, r_10, l_20, r_20, calib, 10.0, aruco_3d, needle_tip)


if __name__ == "__main__":
    # testset1()
    main()
