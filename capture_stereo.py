import cv2
import numpy as np
import time
import inspect
import sys

from helpers import *

import configs as Cfg


def _capture_frames_stereo_with_charuco_board(
    detector,
    board,
    save_dir,
    num_frames,
    cam_left,
    cam_right,
    min_common_ids,
    min_move_px,
    cooldown,
    resolution,
):

    mkdir_empty(save_dir)

    cap_l = cv2.VideoCapture(cam_left)
    cap_r = cv2.VideoCapture(cam_right)

    # Mitigate the delay between captures.
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

    valid_left = 0
    valid_right = 0
    valid_stereo = 0
    saved = 0

    last_save_time = 0
    last_centroid = None

    print("▶ AutoCapture activated (press ESC to exit.)\n")

    while saved < num_frames:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        cl, ids_l = detect_charuco(frame_l, detector, board)
        cr, ids_r = detect_charuco(frame_r, detector, board)

        ok_l = cl is not None
        ok_r = cr is not None

        ok_common = False
        count_common_ids = 0

        # --- 1: common ids ---
        if ok_l and ok_r:
            common_ids = np.intersect1d(ids_l.flatten(), ids_r.flatten())
            count_common_ids = len(common_ids)
            ok_common = count_common_ids >= min_common_ids

        # --- 2: movement ---
        centroid = None
        has_moved = True

        if ok_l:
            centroid = np.mean(cl, axis=0).ravel()

            if last_centroid is not None:
                has_moved = np.linalg.norm(centroid - last_centroid) > min_move_px

        # --- 3: time diff with the last photo ---
        now = time.time()
        cooldown_ok = (now - last_save_time) > cooldown

        auto_capture_ok = ok_l and ok_r and ok_common and has_moved and cooldown_ok

        # --- Visualization ---

        vis_l = frame_l.copy()
        vis_r = frame_r.copy()
        if ok_l:
            cv2.aruco.drawDetectedCornersCharuco(vis_l, cl, ids_l)
        if ok_r:
            cv2.aruco.drawDetectedCornersCharuco(vis_r, cr, ids_r)
        color = (0, 255, 0) if auto_capture_ok else (0, 0, 255)
        cv2.putText(
            vis_l,
            f"L:{valid_left} R:{valid_right} S:{valid_stereo}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        cv2.putText(
            vis_l,
            f"Common IDs: {count_common_ids}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        cv2.imshow(cam_left, vis_l)
        cv2.imshow(cam_right, vis_r)

        # ---- AUTO SAVE ----
        if auto_capture_ok:
            cv2.imwrite(f"{save_dir}/left_with_charuco_{saved:02d}.png", frame_l)
            cv2.imwrite(f"{save_dir}/right_with_charuco_{saved:02d}.png", frame_r)
            valid_left += 1
            valid_right += 1
            valid_stereo += 1
            saved += 1
            last_save_time = now
            last_centroid = centroid
            print(f"✔ Auto-capture {saved:02d} | Common IDs: {count_common_ids}")

        if cv2.waitKey(1) == 27:
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


def _capture_frames_stereo_without_markers(
    save_dir, num_frames, cam_left, cam_right, min_move_px, resolution
):

    mkdir_empty(save_dir)
    cap_l = cv2.VideoCapture(cam_left)
    cap_r = cv2.VideoCapture(cam_right)
    # Mitigate the delay between captures.
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

    saved = 0

    print("▶ Press 'c' to capture. Press ESC to exit.)\n")
    saving = False

    while saved < num_frames:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        # --- Visualization ---

        vis_l = frame_l.copy()
        vis_r = frame_r.copy()
        color = (0, 255, 0)
        cv2.putText(
            vis_l, f"Saved: {saved}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
        cv2.imshow(cam_left, vis_l)
        cv2.imshow(cam_right, vis_r)

        if saving:
            cv2.imwrite(f"{save_dir}/left_without_marker_{saved:02d}.png", frame_l)
            cv2.imwrite(f"{save_dir}/right_without_marker_{saved:02d}.png", frame_r)
            saved += 1
            print(f"✔ Saved pair {saved:02d}")
            saving = False

        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord("c"):
            saving = True

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


def add_testset_info(testset_dir, res, baseline, rotation):
    f = open(f"{testset_dir}/testset_info.txt", "w")
    print("resolution = ", res)
    print("baseline = ", baseline)
    print("rotation = ", rotation)
    f.close()


def capture_images_for_testset(testset_dir, res, baseline, rotation):
    detector = cv2.aruco.ArucoDetector(Cfg.ARUCO_DICT, cv2.aruco.DetectorParameters())
    board = cv2.aruco.CharucoBoard(
        (Cfg.CHARUCO_COLS, Cfg.CHARUCO_ROWS),
        Cfg.SQUARE_LENGTH,
        Cfg.MARKER_LENGTH,
        Cfg.ARUCO_DICT,
    )
    if not dir_not_exist_or_empty(testset_dir):
        print("The directory for the testset exists and is not empty")
        exit()

    add_testset_info(testset_dir, res, baseline, rotation)

    print("\n----------------------------------------------------------------")
    print("                      CAPTURING FRAMES                          ")
    print("\n----------------------------------------------------------------")

    _capture_frames_stereo_with_charuco_board(
        detector,
        board,
        save_dir=f"{testset_dir}/images_for_calibration",
        num_frames=50,
        cam_left="/dev/video0",
        cam_right="/dev/video2",
        min_common_ids=12,
        min_move_px=15,
        cooldown=2.0,
        resolution=res,
    )

    print("\n----------------------------------------------------------------")
    print("               CAPTURING IMAGES OF A METER                      ")
    print("\n----------------------------------------------------------------")

    _capture_frames_stereo_without_markers(
        save_dir=f"{testset_dir}/images_for_measuring",
        num_frames=4,
        cam_left="/dev/video0",
        cam_right="/dev/video2",
        min_move_px=15,
        resolution=res,
    )


def main(testset_dir, force_overwrite=False):

    if not dir_not_exist_or_empty(testset_dir) and not force_overwrite:
        print("")
        print("The directory {testset_dir} exists and is not empty. If you")
        print("want to force overwriting it, use the option --remove when")
        print("calling this program")
        print("")
        exit()

    rmdir(testset_dir)

    print(
        """
    1 - 640x480. parallel. baseline small.
    2 - 640x480. parallel. baseline medium.
    3 - 640x480. 30degrees. baseline small.
    4 - 640x480. 30degrees. baseline medium.
    """
    )

    s = "small"
    m = "medium"
    par = "parallel"
    deg30 = "30degrees"
    lowres = (640, 480)
    choice = int(input("\nChoose a function: "))

    if choice == 1:
        capture_images_for_testset(lowres, s, par)
    elif choice == 2:
        capture_images_for_testset(lowres, m, par)
    elif choice == 3:
        capture_images_for_testset(lowres, s, deg30)
    elif choice == 4:
        capture_images_for_testset(lowres, m, deg30)


# def capture_stereo_images_just_testing():
#    _capture_frames_stereo_without_markers( save_dir = "blahblah",
#        num_frames = 3, cam_left = "/dev/video0", cam_right = "/dev/video2",
#        min_move_px = 15, resolution = (640, 480)
#    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("testset_dir", type=str, help="Root path of the testset")
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Force removing the testset_dir if it exists and is not empty",
    )

    args = parser.parse_args()

    main(testset_dir=args.testset_dir, force_overwrite=args.remove)
