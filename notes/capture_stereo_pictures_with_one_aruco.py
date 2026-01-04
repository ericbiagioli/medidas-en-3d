import cv2
import numpy as np
import os
import tkinter as tk

from helpers import show_fitted

ar_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
ar_pars = cv2.aruco.DetectorParameters()
ar_pars.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

detector = cv2.aruco.ArucoDetector(ar_dict, ar_pars)


def aruco_points_and_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None:
        return None, None

    pts = corners[0][0].astype(np.float32)
    center = pts.mean(axis=0)

    return pts, center


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def draw_aruco_border(image, border):
    if image is not None and border is not None:
        corners = np.array(border, dtype=np.float32).reshape(-1, 2)
        corners_i = corners.astype(int)
        cv2.polylines(image, [corners_i], True, (0, 255, 0), 2)


def capture_images(
    calib="stereo_charuco_calibration.npz",
    resolution=(640, 480),
    debug=False,
    save_dir="tests",
    cam_left="/dev/video2",
    cam_right="/dev/video0",
):

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"❌ Error al intentar leer el archivo de calibración: {calib}.")
        print("")
        exit()

    K1 = data["K1"]
    D1 = data["D1"]
    K2 = data["K2"]
    D2 = data["D2"]

    R1 = data["R1"]
    R2 = data["R2"]
    P1 = data["P1"]
    P2 = data["P2"]
    Q = data["Q"]

    if debug:
        print("P1:\n", P1)
        print("P2:\n", P2)
        print("resolution: ", resolution)
        print("Image size assumed by P1: ", P1[0, 2] * 2, P1[1, 2] * 2)

    mkdir(save_dir)

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
    cv2.namedWindow("L_rect", cv2.WINDOW_NORMAL)
    cv2.namedWindow("R_rect", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("L", half_w, half_h)
    cv2.resizeWindow("R", half_w, half_h)
    cv2.resizeWindow("L_rect", half_w, half_h)
    cv2.resizeWindow("R_rect", half_w, half_h)

    cv2.moveWindow("L", 0, 0)
    cv2.moveWindow("R", half_w, 0)
    cv2.moveWindow("L_rect", 0, half_h)
    cv2.moveWindow("R_rect", half_w, half_h)

    # ---------------------
    # Cameras
    # ---------------------

    # cap_l = cv2.VideoCapture("/dev/video2")
    # cap_r = cv2.VideoCapture("/dev/video0")
    cap_l = cv2.VideoCapture(cam_left)
    cap_r = cv2.VideoCapture(cam_right)

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

    # ---------------------
    # Rectification maps
    # ---------------------

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, resolution, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, resolution, cv2.CV_32FC1)

    print("▶ ESC para salir")

    # =====================
    # Loop
    # =====================

    saved_a = 0
    saved_b = 0
    saved_c = 0

    while True:

        ok_L, L = cap_l.read()
        ok_R, R = cap_r.read()

        if not ok_L or not ok_R:
            continue

        cansave = False

        L_rect = cv2.remap(L, map1x, map1y, cv2.INTER_LINEAR)
        R_rect = cv2.remap(R, map2x, map2y, cv2.INTER_LINEAR)

        vis_L = None
        vis_R = None
        vis_L_rect = None
        vis_R_rect = None

        if L is not None:
            vis_L = L.copy()
        if R is not None:
            vis_R = R.copy()
        if L_rect is not None:
            vis_L_rect = L_rect.copy()
        if R_rect is not None:
            vis_R_rect = R_rect.copy()

        ## Aruco points and centers, both in rectified and in original images
        points_L, center_float_L = aruco_points_and_center(L)
        points_R, center_float_R = aruco_points_and_center(R)
        points_L_rect, center_float_L_rect = aruco_points_and_center(L_rect)
        points_R_rect, center_float_R_rect = aruco_points_and_center(R_rect)

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

        if center_int_L is not None:
            cv2.circle(vis_L, center_int_L, 8, (0, 255, 0), -1)
        if center_int_R is not None:
            cv2.circle(vis_R, center_int_R, 8, (0, 255, 0), -1)
        if center_int_L_rect is not None:
            cv2.circle(vis_L_rect, center_int_L_rect, 8, (0, 255, 0), -1)
        if center_int_R_rect is not None:
            cv2.circle(vis_R_rect, center_int_R_rect, 8, (0, 255, 0), -1)

        draw_aruco_border(vis_L, points_L)
        draw_aruco_border(vis_R, points_R)
        draw_aruco_border(vis_L_rect, points_L_rect)
        draw_aruco_border(vis_R_rect, points_R_rect)

        if vis_L is None:
            vis_L = np.zeros(resolution)
        if vis_R is None:
            vis_R = np.zeros(resolution)
        if vis_L_rect is None:
            vis_L_rect = np.zeros(resolution)
        if vis_R_rect is None:
            vis_R_rect = np.zeros(resolution)

        cv2.putText(
            vis_L,
            f"a = {saved_a}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            vis_L,
            f"b = {saved_b}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            vis_L,
            f"c = {saved_c}",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2,
        )

        show_fitted("L", vis_L)
        show_fitted("R", vis_R)
        show_fitted("L_rect", vis_L_rect)
        show_fitted("R_rect", vis_R_rect)

        if (
            center_float_L is not None
            and center_float_R is not None
            and center_float_L_rect is not None
            and center_float_R_rect is not None
        ):
            cansave = True

        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord("a") or k == ord("b") or k == ord("c"):
            if not cansave:
                continue

            if k == ord("a"):
                cv2.imwrite(f"{save_dir}/l_20.0_{saved_a:02d}.png", L)
                cv2.imwrite(f"{save_dir}/r_20.0_{saved_a:02d}.png", R)
                saved_a = saved_a + 1
            elif k == ord("b"):
                cv2.imwrite(f"{save_dir}/l_10.0_{saved_b:02d}.png", L)
                cv2.imwrite(f"{save_dir}/r_10.0_{saved_b:02d}.png", R)
                saved_b = saved_b + 1
            elif k == ord("c"):
                cv2.imwrite(f"{save_dir}/l_1.0_{saved_c:02d}.png", L)
                cv2.imwrite(f"{save_dir}/r_1.0_{saved_c:02d}.png", R)
                saved_c = saved_c + 1
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


def main():
    print(
        """

Este script toma fotos para testear un par estéreo. Se asume que en las fotos
tomadas pulsando 'a', la punta de la aguja está en el marcador 20.0 del
instrumento de medición. En las fotos tomadas pulsando 'b', la punta de la
aguja está en la posición 10.0 del instrumento de medición. En las fotos
tomadas pulsando 'c', la punta de la aguja está en la posición 1.0 del
instrumento de medición.

"""
    )
    capture_images()


if __name__ == "__main__":
    main()
