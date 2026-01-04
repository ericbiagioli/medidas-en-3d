import cv2
import numpy as np
import tkinter as tk
from types import SimpleNamespace
import os

from helpers import show_fitted


# =========================
# Load stereo calibration
# =========================

data = np.load("stereo_charuco_calibration.npz")

K1 = data["K1"]
D1 = data["D1"]
K2 = data["K2"]
D2 = data["D2"]

R1 = data["R1"]
R2 = data["R2"]
P1 = data["P1"]
P2 = data["P2"]
Q = data["Q"]


ARUCO_SIZE = 0.043
aruco_3d = np.array(
    [
        [-ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],
        [ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],
        [ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
        [-ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
    ],
    dtype=np.float32,
)
# tip_aruco = np.array([-0.040, 0.145, -0.003])  # ejemplo
tip_aruco = np.array([-0.025, 0.0, 0.00])  # ejemplo


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


def triangulate_point_rectified(pt_l, pt_r):
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


ARUCO_SIZE = 0.043
aruco_3d = np.array(
    [
        [-ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],
        [ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],
        [ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
        [-ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
    ],
    dtype=np.float32,
)
# tip_aruco = np.array([-0.040, 0.145, -0.003])  # ejemplo
tip_aruco = np.array([-0.025, -0.08, 0.003])  # ejemplo


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


def main(resolution=(640, 480)):

    print("P1:\n", P1)
    print("P2:\n", P2)
    print("Image size assumed by P1:", P1[0, 2] * 2, P1[1, 2] * 2)

    TESTS_DIR = "tests"
    mkdir(TESTS_DIR)

    # ---------------------
    # Windows
    # ---------------------

    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    half_w = screen_w // 2
    half_h = screen_h // 2

    cv2.namedWindow("L", cv2.WINDOW_NORMAL)
    cv2.namedWindow("R", cv2.WINDOW_NORMAL)
    cv2.namedWindow("L-rect", cv2.WINDOW_NORMAL)
    cv2.namedWindow("R-rect", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("L", half_w, half_h)
    cv2.resizeWindow("R", half_w, half_h)
    cv2.resizeWindow("L-rect", half_w, half_h)
    cv2.resizeWindow("R-rect", half_w, half_h)

    cv2.moveWindow("L", 0, 0)
    cv2.moveWindow("R", half_w, 0)
    cv2.moveWindow("L-rect", 0, half_h)
    cv2.moveWindow("R-rect", half_w, half_h)

    # ---------------------
    # Cameras
    # ---------------------

    cap_l = cv2.VideoCapture("/dev/video2")
    cap_r = cv2.VideoCapture("/dev/video0")

    cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    h, w = resolution[1], resolution[0]

    # ---------------------
    # Rectification maps
    # ---------------------

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, resolution, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, resolution, cv2.CV_32FC1)

    print("▶ ESC para salir")

    # =====================
    # Loop
    # =====================

    X1 = None
    X2 = None
    dist = None
    saved_a = 1
    saved_b = 1
    saved_c = 1
    while True:

        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()

        if not ok_l or not ok_r:
            continue

        cansave = False

        l_rect = cv2.remap(frame_l, map1x, map1y, cv2.INTER_LINEAR)
        r_rect = cv2.remap(frame_r, map2x, map2y, cv2.INTER_LINEAR)

        ## pts_l

        pts_l, cl_float = aruco_points_and_center(frame_l)
        pts_r, cr_float = aruco_points_and_center(frame_r)

        pts_l_rect, cl_rect_float = aruco_points_and_center(l_rect)
        pts_r_rect, cr_rect_float = aruco_points_and_center(r_rect)

        ##print(pts_l)

        if cl_float is not None and cr_float is not None:
            cl = tuple(np.round(cl_float).astype(int))
            cr = tuple(np.round(cr_float).astype(int))

            cv2.circle(frame_l, cl, 5, (0, 255, 0), -1)
            cv2.circle(frame_r, cr, 5, (0, 255, 0), -1)

            c3d = cv2.triangulatePoints(
                P1, P2, cl_float.reshape(2, 1), cr_float.reshape(2, 1)
            )
            c3d_flat = (c3d[:3] / c3d[3]).flatten()  # ahora sí (3,)
            reproy_l = project_with_P(P1, c3d_flat)
            reproy_r = project_with_P(P2, c3d_flat)
            reproy_l = tuple(np.round(reproy_l).astype(int))
            reproy_r = tuple(np.round(reproy_r).astype(int))

            cv2.circle(frame_l, reproy_l, 5, (255, 255, 0), -1)
            cv2.circle(frame_r, reproy_r, 5, (255, 255, 0), -1)

        if cl_rect_float is not None and cr_rect_float is not None:

            ##print("y_L_rect used:", cl_rect_float[1])
            ##print("image height:", l_rect.shape[0])

            ## Centro en imagen rectificada

            cl_rect = tuple(np.round(cl_rect_float).astype(int))
            cr_rect = tuple(np.round(cr_rect_float).astype(int))

            cv2.circle(l_rect, cl_rect, 20, (0, 255, 0), -1)
            cv2.circle(r_rect, cr_rect, 20, (0, 255, 0), -1)

            ## Triangulación y reproyección del centro

            c3d_rect = cv2.triangulatePoints(
                P1, P2, cl_rect_float.reshape(2, 1), cr_rect_float.reshape(2, 1)
            )
            c3d_rect_flat = (c3d_rect[:3] / c3d_rect[3]).flatten()  # ahora sí (3,)
            reproy_l_rect = project_with_P(P1, c3d_rect_flat)
            reproy_r_rect = project_with_P(P2, c3d_rect_flat)
            reproy_l_rect = tuple(np.round(reproy_l_rect).astype(int))
            reproy_r_rect = tuple(np.round(reproy_r_rect).astype(int))

            cv2.circle(l_rect, reproy_l_rect, 10, (255, 0, 0), -1)
            cv2.circle(r_rect, reproy_r_rect, 10, (255, 0, 0), -1)

            ## Utilizar disparidad en lugar de triangular

            disp = cl_rect_float[0] - cr_rect_float[0]
            vec = np.array([cl_rect_float[0], cl_rect_float[1], disp, 1.0])
            Xqh = Q @ vec
            Xq = Xqh[:3] / Xqh[3]
            reproj_using_Q_l = project_with_P(P1, Xq)
            reproj_using_Q_l = tuple(np.round(reproj_using_Q_l).astype(int))

            cv2.circle(l_rect, reproj_using_Q_l, 5, (0, 0, 255), -1)

            ## dibujar la punta del lapiz
            X_corners = []
            for i in range(4):
                X = triangulate_point_rectified(pts_l_rect[i], pts_r_rect[i])
                X_corners.append(X)
            X_corners = np.array(X_corners)
            R_aruco, t_aruco = rigid_alignment(aruco_3d, X_corners)
            X_tip = R_aruco @ tip_aruco + t_aruco
            pL_tip = project_with_P(P1, X_tip)
            pR_tip = project_with_P(P2, X_tip)
            pL_tip = tuple(np.round(pL_tip).astype(int))
            pR_tip = tuple(np.round(pR_tip).astype(int))
            cv2.circle(l_rect, pL_tip, 6, (0, 255, 255), -1)

        show_fitted("L", frame_l)
        show_fitted("R", frame_r)
        show_fitted("L-rect", l_rect)
        show_fitted("R-rect", r_rect)

        if (
            cl_float is not None
            and cr_float is not None
            and cl_rect_float is not None
            and cr_rect_float is not None
        ):
            cansave = True

        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord("a") or k == ord("b") or k == ord("c"):
            print("Se apreto la tecla C")
            if X1 is None:
                X1 = X_tip
            elif X2 is None:
                X2 = X_tip
                dist = np.linalg.norm(X2 - X1)
                if cansave == False:
                    print("Cannot save. Invalid frames")
                    continue
                if k == ord("a"):
                    cv2.imwrite(f"{TESTS_DIR}/left_pos=20.0_{saved_a:02d}.png", frame_l)
                    cv2.imwrite(
                        f"{TESTS_DIR}/right_pos=20.0_{saved_a:02d}.png", frame_r
                    )
                    saved_a = saved_a + 1
                elif k == ord("b"):
                    cv2.imwrite(f"{TESTS_DIR}/left_pos=10.0_{saved_b:02d}.png", frame_l)
                    cv2.imwrite(
                        f"{TESTS_DIR}/right_pos=10.0_{saved_b:02d}.png", frame_r
                    )
                    saved_b = saved_b + 1
                elif k == ord("c"):
                    cv2.imwrite(f"{TESTS_DIR}/left_pos=1.0_{saved_c:02d}.png", frame_l)
                    cv2.imwrite(f"{TESTS_DIR}/right_pos=1.0_{saved_c:02d}.png", frame_r)
                    saved_c = saved_c + 1
            else:
                X1 = X2
                X2 = X_tip
                print("Seteando ambos")
                print("calculando dist...")
                dist = np.linalg.norm(X2 - X1)
                print("dist = ", dist)
                if cansave == False:
                    print("Cannot save. Invalid frames")
                    continue
                if k == ord("a"):
                    cv2.imwrite(f"{TESTS_DIR}/left_pos=20.0_{saved_a:02d}.png", frame_l)
                    cv2.imwrite(
                        f"{TESTS_DIR}/right_pos=20.0_{saved_a:02d}.png", frame_r
                    )
                    saved_a = saved_a + 1
                    print(
                        f"saved_a = {saved_a-1}, saved_b = {saved_b-1}, saved_c = {saved_c-1}"
                    )
                elif k == ord("b"):
                    cv2.imwrite(f"{TESTS_DIR}/left_pos=10.0_{saved_b:02d}.png", frame_l)
                    cv2.imwrite(
                        f"{TESTS_DIR}/right_pos=10.0_{saved_b:02d}.png", frame_r
                    )
                    saved_b = saved_b + 1
                    print(
                        f"saved_a = {saved_a-1}, saved_b = {saved_b-1}, saved_c = {saved_c-1}"
                    )
                elif k == ord("c"):
                    cv2.imwrite(f"{TESTS_DIR}/left_pos=1.0_{saved_c:02d}.png", frame_l)
                    cv2.imwrite(f"{TESTS_DIR}/right_pos=1.0_{saved_c:02d}.png", frame_r)
                    saved_c = saved_c + 1
                    print(
                        f"saved_a = {saved_a-1}, saved_b = {saved_b-1}, saved_c = {saved_c-1}"
                    )
            print("X1 = ", X1, "  X2 = ", X2, "dist = ", dist)

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main((640, 480))
