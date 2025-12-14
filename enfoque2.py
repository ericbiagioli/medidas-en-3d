import cv2
import torch
from deep_aruco.detector import MarkerDetector
from deep_aruco.corner_refiner import CornerRefiner
from deep_aruco.decoder import MarkerDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar los modelos
detector = MarkerDetector("models/detector.pth").to(device)
refiner = CornerRefiner("models/corners.pth").to(device)
decoder = MarkerDecoder("models/decoder.pth").to(device)

detector.eval()
refiner.eval()
decoder.eval()

# Abrir webcam o imagen
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. DETECCIÓN (Bounding boxes o regiones)
    with torch.no_grad():
        detections = detector.detect(img_rgb, device=device)
        # detections = lista de regiones con formato (x1, y1, x2, y2)

    # 2. PARA CADA MARCADOR DETECTADO → refinamiento de esquinas
    for det in detections:
        x1, y1, x2, y2 = det

        roi = img_rgb[y1:y2, x1:x2]

        with torch.no_grad():
            corners = refiner.refine(roi, device=device)
            # corners → arreglo Nx2 con las esquinas refinadas dentro del ROI

        # Convertir esquinas del ROI a coords globales
        corners[:, 0] += x1
        corners[:, 1] += y1

        # 3. DECODIFICACIÓN DEL ID
        with torch.no_grad():
            marker_id = decoder.decode(img_rgb, corners, device=device)

        # Dibujo del resultado
        for cx, cy in corners:
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)

        cv2.putText(frame, f"ID {marker_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    cv2.imshow("DeepArUco++", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


