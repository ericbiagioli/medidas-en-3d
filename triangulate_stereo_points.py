import cv2
import numpy as np
from itertools import combinations


def rectify_point(x, y, K, D, R, P):
    pts = np.array([[[x, y]]], dtype=np.float64)
    pts_rect = cv2.undistortPoints(pts, K, D, R=R, P=P)
    return pts_rect[0, 0, 0], pts_rect[0, 0, 1]


def main():
    image_size = (640, 480)
    calib = "testsets/640x480-baseline-small/calibration.npz"

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

    img_size = (640, 480)

    # Rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

    points = {
        "30": ((205, 185), (469, 156)),
        "32": ((204, 248), (466, 220)),
        "34": ((204, 309), (462, 283)),
    }

    points_3d_1 = {}  ## Metodo 1
    points_3d_2 = {}  ## Metodo 2

    ## Metodo 1
    def reproject_point(x, y, d, Q):
        point = np.array([x, y, d, 1.0])
        X = Q @ point
        X /= X[3]
        return X[:3]  # (X, Y, Z)

    for label, ((x1, y1), (x2, y2)) in points.items():
        x1r, y1r = rectify_point(x1, y1, K1, D1, R1, P1)
        x2r, y2r = rectify_point(x2, y2, K2, D2, R2, P2)

        if abs(y1r - y2r) > 0.5:
            print(f"Error epipolar en {label}: {y1r - y2r}")

        ## Metodo 1: disparidad
        d = x1r - x2r
        P = reproject_point(x1r, y1r, d, Q)
        points_3d_1[label] = P
        print(f"Points3D (metodo1) {label} = {P}")

        ## Metodo 2: triangulacion explicita
        pts1 = np.array([[x1r, y1r]], dtype=np.float64).T
        pts2 = np.array([[x2r, y2r]], dtype=np.float64).T

        X_h = cv2.triangulatePoints(P1, P2, pts1, pts2)
        X = (X_h / X_h[3])[:3].reshape(-1)
        points_3d_2[label] = X

    print("Metodo 1:")
    for l1, l2 in combinations(points_3d_1.keys(), 2):
        p1 = points_3d_1[l1]
        p2 = points_3d_1[l2]
        dist = np.linalg.norm(p1 - p2)
        print(f"Distancia {l1} – {l2}: {dist:.4f}")

    print("Metodo 2:")
    for l1, l2 in combinations(points_3d_2.keys(), 2):
        p1 = points_3d_2[l1]
        p2 = points_3d_2[l2]
        dist = np.linalg.norm(p1 - p2)
        print(f"Distancia {l1} – {l2}: {dist:.4f}")


if __name__ == "__main__":
    main()
