import cv2
import numpy as np
import glob

from calibrate_mono import calibrate_mono
from helpers import *

import configs as Cfg


def validate_image_size(l_images, r_images):
    image_size = None

    for fname_l, fname_r in zip(l_images, r_images):
        img_l = cv2.imread(fname_l)
        img_r = cv2.imread(fname_r)

    if img_l is None or img_r is None:
        raise RuntimeError("Cannot read some of the images.")

    if image_size is None:
        image_size = (img_l.shape[1], img_l.shape[0])
    else:
        if (img_l.shape[1], img_l.shape[0]) != image_size:
            raise RuntimeError("Not all LEFT images have the same size.")
        if (img_r.shape[1], img_r.shape[0]) != image_size:
            raise RuntimeError("Not all RIGHT images have the same size.")

    return image_size


def calibrate_stereo(testset_dir):

    expression_l = "images_for_calibration/left_with_charuco_*.png"
    expression_r = "images_for_calibration/right_with_charuco_*.png"

    fn_output = f"{testset_dir}/calibration_output.txt"
    fn_calibration = f"{testset_dir}/calibration.npz"

    remove(fn_output)
    remove(fn_calibration)

    detector = cv2.aruco.ArucoDetector(Cfg.ARUCO_DICT, cv2.aruco.DetectorParameters())
    board = cv2.aruco.CharucoBoard(
        (Cfg.CHARUCO_COLS, Cfg.CHARUCO_ROWS),
        Cfg.SQUARE_LENGTH,
        Cfg.MARKER_LENGTH,
        Cfg.ARUCO_DICT,
    )

    left_images = sorted(glob.glob(f"{testset_dir}/{expression_l}"))
    right_images = sorted(glob.glob(f"{testset_dir}/{expression_r}"))

    image_size = validate_image_size(left_images, right_images)
    if image_size is None:
        raise RuntimeError("Error computing the image size.")

    print_tee("\n▶ Calibrating left camera.", fn=fn_output)
    rms_left, K1, D1, _ = calibrate_mono(left_images, detector, board)
    print_tee("RMS LEFT =", rms_left, fn=fn_output)

    print_tee("\n▶ Calibrating right camera.", fn=fn_output)
    rms_right, K2, D2, _ = calibrate_mono(right_images, detector, board)
    print_tee("RMS LEFT =", rms_right, fn=fn_output)

    print_tee("\n▶ Calibrating stereo pair.", fn=fn_output)
    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    for lf, rf in zip(left_images, right_images):
        img_l = cv2.imread(lf)
        img_r = cv2.imread(rf)

        cl, il = detect_charuco(img_l, detector, board)
        cr, ir = detect_charuco(img_r, detector, board)

        if cl is None or cr is None:
            continue

        common_ids = np.intersect1d(il.flatten(), ir.flatten())

        if len(common_ids) < 6:
            continue

        obj = []
        pts_l = []
        pts_r = []

        chessboardCorners = board.getChessboardCorners()
        for cid in common_ids:
            idx_l = np.where(il == cid)[0][0]
            idx_r = np.where(ir == cid)[0][0]

            obj.append(chessboardCorners[cid])
            pts_l.append(cl[idx_l])
            pts_r.append(cr[idx_r])

        objpoints.append(np.array(obj, dtype=np.float32))
        imgpoints_l.append(np.array(pts_l, dtype=np.float32))
        imgpoints_r.append(np.array(pts_r, dtype=np.float32))

    if len(objpoints) == 0:
        raise RuntimeError(
            "It was not posible to detect valid stereo views in the images."
        )

    flags = cv2.CALIB_FIX_INTRINSIC

    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, K1, D1, K2, D2, image_size, flags=flags
    )

    print_tee("RMS STEREO PAIR =", rms, fn=fn_output)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    np.savez(
        fn_calibration,
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        R=R,
        T=T,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
    )

    print_tee("\nCalibration process completed.", fn=fn_output)
    print_tee("P1 =\n", P1, fn=fn_output)
    print_tee("P2 =\n", P2, fn=fn_output)

    return rms, R, T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("testset_dir", type=str, help="Root path of the testset")
    args = parser.parse_args()

    calibrate_stereo(args.testset_dir)
