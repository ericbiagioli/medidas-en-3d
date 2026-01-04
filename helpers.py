import glob
import shutil

import cv2
import numpy as np
import os


def remove(fn):
    if os.path.exists(fn):
        os.remove(fn)


def rmdir(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdir_empty(path):

    if not os.path.exists(path):
        os.makedirs(path)

    for p in glob.glob(os.path.join(path, "*")):
        if os.path.isfile(p) or os.path.islink(p):
            os.remove(p)


def dir_not_exist_or_empty(path):
    if not os.path.exists(path):
        return True
    if os.path.isdir(path) and not os.listdir(path):
        return True
    return False


def print_tee(*args, fn, **kwargs):
    f = open(fn, "a")
    print(*args, **kwargs)
    print(*args, **kwargs, file=f)
    f.close()


def show_fitted(win, img):
    img_h, img_w = img.shape[:2]

    _, _, win_w, win_h = cv2.getWindowImageRect(win)

    scale = min(win_w / img_w, win_h / img_h)

    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    cv2.imshow(win, resized)


def distance(p1, p2):
    if p1 is None or p2 is None:
        return None

    if isinstance(p1, str) and p1 == "capturing":
        return None

    if isinstance(p2, str) and p2 == "capturing":
        return None

    p1 = np.asarray(p1, dtype=np.float64).reshape(-1)
    p2 = np.asarray(p2, dtype=np.float64).reshape(-1)

    if p1.shape != (3,) or p2.shape != (3,):
        return None

    return np.linalg.norm(p1 - p2)


def detect_charuco(frame, detector, board):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, None

    _, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )

    if char_ids is None or len(char_ids) < 6:
        return None, None

    return char_corners, char_ids
