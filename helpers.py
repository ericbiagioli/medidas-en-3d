import cv2
import numpy as np

"""
Ajusta el tamaño de la imagen `img` para que entre en la ventana `win` y
la muestra.
"""
def show_fitted(win, img):
    img_h, img_w = img.shape[:2]

    _, _, win_w, win_h = cv2.getWindowImageRect(win)

    scale = min(win_w / img_w, win_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    cv2.imshow(win, resized)


"""
Calcula la matriz de cámara y las distorsiones radiales y tangenciales. De
momento, propongo una aproximación bien simple.

ASUMO QUE el punto principal (esto es: la intersección del eje óptico con
el sensor) está ubicado exactamente en el centro geométrico del sensor. Esto
es: (w / 2, h / 2). Esto raramente es así; pero de momento lo asumo.
@TODO: Analizar si esta suposición afecta a las mediciones. Podría ser que en
realidad no importe para nuestros fines si el punto principal no es el que
consideramos.

ASUMO QUE la distancia focal es 0.8 * max(w, h).

ASUMO QUE no hay distorsión (ni radial ni tangencial).
"""
def default_camera_matrix(w: int, h: int):
    focal_length_px = 0.8 * max(w, h)
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[focal_length_px, 0, cx],
                  [0, focal_length_px, cy],
                  [0, 0, 1]], dtype=np.float64)

    distorsion = np.zeros((5, 1), dtype=np.float64)

    return K, distorsion


def draw_results(image, results, K, D, marker_length_m):
    out = image.copy()
    for r in results:
        # borde
        corners = np.array(r['corners'], dtype=np.float32).reshape(-1, 2)
        corners_i = corners.astype(int)
        cv2.polylines(out, [corners_i], True, (0, 255, 0), 2)

        # ejes
        cv2.drawFrameAxes(out, K, D, r['rvec'], r['tvec'], marker_length_m * 0.5)

    return out

def get_distance(p1, p2):
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

