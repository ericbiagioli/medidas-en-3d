from helpers import *


def validate_image_size(images):
    image_size = None

    for fname in images:
        img = cv2.imread(fname)

        if img is None:
            raise RuntimeError("Cannot read some of the images.")

        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        else:
            if (img.shape[1], img.shape[0]) != image_size:
                raise RuntimeError("Not all images have the same size.")

    return image_size


def calibrate_mono(images, detector, board):

    all_corners = []
    all_ids = []

    image_size = validate_image_size(images)
    if image_size is None:
        raise RuntimeError("Error computing the image size.")

    for fname in images:
        img = cv2.imread(fname)

        corners, ids = detect_charuco(img, detector, board)
        if corners is not None:
            all_corners.append(corners)
            all_ids.append(ids)

    # --- Safe initialization of K ---
    fx = fy = 0.8 * image_size[0]
    cx = image_size[0] / 2
    cy = image_size[1] / 2

    K_init = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    D_init = np.zeros((5, 1))

    rms, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners,
        all_ids,
        board,
        image_size,
        K_init,
        D_init,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )

    if len(all_corners) == 0:
        raise RuntimeError(
            "It was not posible to detect valid ChArUco corners in the images."
        )

    return rms, K, D, image_size
