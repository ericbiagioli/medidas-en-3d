import cv2
import numpy as np
from itertools import combinations


def rectify_point(x, y, K, D, R, P):
    pts = np.array([[[x, y]]], dtype=np.float64)
    pts_rect = cv2.undistortPoints(pts, K, D, R=R, P=P)
    return pts_rect[0, 0, 0], pts_rect[0, 0, 1]


def main():
    image_size = (640, 480)

    testset_dir = "testsets/640x480_baseline_small_parallel__0"
    calib = f"{testset_dir}/calibration.npz"

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

    points_coords = open(f"{testset_dir}/detected_coordinates.txt", "r")
    code = points_coords.read()
    points_coords.close()
    ns = {}
    exec(code, ns)
    left_images = ns["left_images"]
    right_images = ns["right_images"]

    assert left_images is not None, "left_images is None"
    assert right_images is not None, "right_images is None"
    assert len(left_images) == len(
        right_images
    ), "left_images and right_images have different length"
    for i in range(len(left_images)):
        assert set(left_images[i].keys()) == set(
            right_images[i].keys()
        ), "some objects don't match the keys"

    # Reprojection
    def estimate_3d_method_1(l_x, l_y, r_x, r_y):
        def reproject_point(x, y, d, Q):
            point = np.array([x, y, d, 1.0])
            X = Q @ point
            X /= X[3]
            return X[:3]

        # epipolar_error_threshold = 0.5
        epipolar_error_threshold = 2.0

        l_x_rect, l_y_rect = rectify_point(l_x, l_y, K1, D1, R1, P1)
        r_x_rect, r_y_rect = rectify_point(r_x, r_y, K2, D2, R2, P2)
        if abs(l_y_rect - r_y_rect) > epipolar_error_threshold:
            print(
                f"Epipolar error = {l_y_rect - r_y_rect}. Bad correspondence or bad calibration!"
            )

        disparity = l_x_rect - r_x_rect

        return reproject_point(l_x_rect, l_y_rect, disparity, Q)

    # Explicit triangulation
    def estimate_3d_method_2(l_x, l_y, r_x, r_y):

        # epipolar_error_threshold = 0.5
        epipolar_error_threshold = 2.0

        l_x_rect, l_y_rect = rectify_point(l_x, l_y, K1, D1, R1, P1)
        r_x_rect, r_y_rect = rectify_point(r_x, r_y, K2, D2, R2, P2)

        if abs(l_y_rect - r_y_rect) > epipolar_error_threshold:
            print(
                f"Epipolar error = {l_y_rect - r_y_rect}. Bad correspondence or bad calibration!"
            )

        pts_l = np.array([[l_x_rect, l_y_rect]], dtype=np.float64).T
        pts_r = np.array([[r_x_rect, r_y_rect]], dtype=np.float64).T

        X_h = cv2.triangulatePoints(P1, P2, pts_l, pts_r)
        return (X_h / X_h[3])[:3].reshape(-1)

    for pair_index in range(len(left_images)):
        keys = list(left_images[pair_index].keys())
        print("Performing measures in the image: ", pair_index)
        for i1 in range(len(keys) - 1):
            l_x1, l_y1 = left_images[pair_index][keys[i1]]
            r_x1, r_y1 = right_images[pair_index][keys[i1]]
            p1_method1 = estimate_3d_method_1(l_x1, l_y1, r_x1, r_y1)
            p1_method2 = estimate_3d_method_2(l_x1, l_y1, r_x1, r_y1)

            for i2 in range(i1 + 1, len(keys)):
                l_x2, l_y2 = left_images[pair_index][keys[i2]]
                r_x2, r_y2 = right_images[pair_index][keys[i2]]
                p2_method1 = estimate_3d_method_1(l_x2, l_y2, r_x2, r_y2)
                p2_method2 = estimate_3d_method_2(l_x2, l_y2, r_x2, r_y2)

                expected = abs(keys[i1] - keys[i2]) / 100.0
                detected_method1 = np.linalg.norm(p1_method1 - p2_method1)
                detected_method2 = np.linalg.norm(p1_method2 - p2_method2)

                print(f"{keys[i1]} -- {keys[i2]}. ", end="")
                print(f"Expected = {expected}. ", end="")
                print(f"Method 1: {detected_method1}. ", end="")
                print(f"Method 2: {detected_method2}")


if __name__ == "__main__":
    main()
