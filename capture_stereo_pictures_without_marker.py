import cv2
import numpy as np
import os
import tkinter as tk

from helpers import show_fitted

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture_images(calib="stereo_charuco_calibration.npz",
    resolution=(640, 480),
    debug=False,
    save_dir="tests",
    cam_left="/dev/video2",
    cam_right="/dev/video0"):

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

    cap_l.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap_l.set(cv2.CAP_PROP_FPS, 30)
    cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    cap_r.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
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

    saved = 0

    while True:

        ok_L, L = cap_l.read()
        ok_R, R = cap_r.read()

        if not ok_L or not ok_R:
            continue

        L_rect = cv2.remap(L, map1x, map1y, cv2.INTER_LINEAR)
        R_rect = cv2.remap(R, map2x, map2y, cv2.INTER_LINEAR)

        vis_L = None
        vis_R = None
        vis_L_rect = None
        vis_R_rect = None

        if L is not None: vis_L = L.copy()
        if R is not None: vis_R = R.copy()
        if L_rect is not None: vis_L_rect = L_rect.copy()
        if R_rect is not None: vis_R_rect = R_rect.copy()

        if vis_L is None: vis_L = np.zeros(resolution)
        if vis_R is None: vis_R = np.zeros(resolution)
        if vis_L_rect is None: vis_L_rect = np.zeros(resolution)
        if vis_R_rect is None: vis_R_rect = np.zeros(resolution)

        cv2.putText(vis_L, f"Saved: {saved}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

        show_fitted("L", vis_L)
        show_fitted("R", vis_R)
        show_fitted("L_rect", vis_L_rect)
        show_fitted("R_rect", vis_R_rect)

        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite(f"{save_dir}/l_{saved:02d}.png", L)
            cv2.imwrite(f"{save_dir}/r_{saved:02d}.png", R)
            saved = saved + 1

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

def main():
    print("""

Este script toma fotos para testear un par estéreo. Se toman las fotos pulsando
la tecla 's'.

""")
    capture_images()

if __name__ == "__main__":
    main()

