def list_cameras(max_devices=10):
    available = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def find_resolutions(device):
    resols = [
        (640, 480),
        (800, 600),
        (1280, 720),
        (1920, 1080),
        (2304, 1536),
    ]

    retval = []
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"Cannot open {device}.")
        return retval

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for w, h in resols:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ret, frame = cap.read()
        if ret:
            retval.append((frame.shape[1], frame.shape[0]))

    cap.release()
    return retval
