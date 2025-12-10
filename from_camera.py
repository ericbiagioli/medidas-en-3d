import argparse
import json
import sys
from pathlib import Path
import tkinter as tk
import cv2
import numpy as np
from types import SimpleNamespace



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
@TODO: mejorar esto.
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

"""
Detecta marcadores ArUco y estima la pose para cada marcador detectado.

ASUMO QUE LOS ARUCOS SIENDO USADOS SON DEL DICCIONARIO 5X5_1000

Parámetros:

- img: imágen BGR (array numpy).
- marker_side_m: tamaño del lado del marcador, en metros.
- camera_matrix: matriz de cámara. Es una matriz numpy de 3x3, calculado o bien
                 con default_camera_matrix o bien con algun proceso de
                 calibración de cámara que permita mejores coeficientes.
- dist_coeffs: Coeficientes de distorsión. Es un array de 5x1, calculado o bien
                 con default_camera_matrix o bien con algun proceso de
                 calibración de cámara que permita mejores coeficientes.




Valor de retorno:

Lista de diccionarios de la forma: [{
                                      'id': int,
                                      'corners': [...],
                                      'rvec': ,
                                      `tvec`:
                                    }, ...]

Donde
  'id' es el valor numérico del ArUco,
  Cada tvec es la posición del centro del marcador en coordenadas de cámara.
  Cada rvec es la orientación del marcador como un vector de rotación (Rodrigues).

  Las coordenadas de cámara son así:
    Eje X de la cámara: hacia la derecha en la imagen.
    Eje Y de la cámara: hacia abajo en la imagen.
    Eje Z de la cámara: hacia adelante desde el foco (positivo hacia la escena).

"""
def estimate_poses_and_corners( img,
                                marker_side_m: float,
                                camera_matrix,
                                dist_coeffs):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

    params = cv2.aruco.DetectorParameters()

    corners_list, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    results = []
    if ids is None or len(ids) == 0:
        return results

    ### Stability
    half = marker_side_m / 2.0

    # Orden: igual al de detectMarkers (TL, TR, BR, BL)
    object_points = np.array([
      [-half,  half, 0],   # top-left
      [ half,  half, 0],   # top-right
      [ half, -half, 0],   # bottom-right
      [-half, -half, 0]    # bottom-left
    ], dtype=np.float32)



    for i, marker_id_found in enumerate(ids.flatten()):
      c = corners_list[i]
      img_pts = c[0].astype(np.float32)  # (4,2)
      ok, rvec, tvec = cv2.solvePnP(object_points, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
      corners = corners_list[i].reshape(-1, 2).tolist()
      results.append({'id': int(marker_id_found),
                       'corners': corners,
                       'rvec':rvec,
                       'tvec':tvec})


    return results


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


"""

                                      MAIN

"""

def process_photo(params):
  filename = "photos/9-49.9.jpg"
  frame = cv2.imread(filename)
  h, w = frame.shape[:2]
  K, D = default_camera_matrix(w, h)
  poses_and_corners = estimate_poses_and_corners(frame, params.marker_side_m, K, D)
  out = draw_results(frame, poses_and_corners, K, D, params.marker_side_m)
  while True:
    show_fitted(params.winname, out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


def process_video(params):
  device = 1
  cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

  if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

  ret, frame = cap.read()
  if not ret:
    print("No se pudo leer un frame de la cámara")
    exit()

  h, w = frame.shape[:2]
  K, D = default_camera_matrix(w, h)

  point1 = None
  point2 = None
  distance = None

  while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer el frame de la cámara")
        exit(0)

    positions_and_corners = estimate_poses_and_corners(frame, params.marker_side_m, K, D)

    if len(positions_and_corners) > 0:
      ## Sistema de coordenadas del marcador: El marcador tiene origen en su centro.
      #+x: Apunta hacia la derecha en la imagen del marcador.
      #+y: Apunta hacia abajo en la imagen del marcador.
      #+z: Apunta perpendicular al plano del marcador, hacia la cámara cuando el marcador está frente a la cámara.
      rvec = positions_and_corners[0]['rvec']
      ## cv2.Rodrigues transforma una representacion vectorial de una rotacion en
      ## la representación matricial, que llamaremos R. La necesitaremos para
      ## calcular donde está la punta del palito.
      R, _ = cv2.Rodrigues(rvec)
      tvec = positions_and_corners[0]['tvec'].reshape(3, 1)
      ## El palito tiene 0.5 cm en x y 32.6 cm en y
      punta_de_palito_en_coordenadas_aruco = np.array([0.01, -0.24, 0.0])
      punta_del_palito_en_coordenadas_de_camara = R @ punta_de_palito_en_coordenadas_aruco.reshape(3,1) + tvec
      centro_del_aruco = tvec


    out = draw_results(frame, positions_and_corners, K, D, params.marker_side_m)

    if len(positions_and_corners) > 0:
      cv2.putText(out, f"Aruco: {centro_del_aruco}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
      cv2.putText(out, f"Palito: {punta_del_palito_en_coordenadas_de_camara}",    (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

      ## Dibujar la punta del palito
      p_mark = punta_de_palito_en_coordenadas_aruco.astype(np.float32).reshape(1,1,3)
      palito_2d, _ = cv2.projectPoints(
        p_mark,
        rvec.astype(np.float32),
        tvec.astype(np.float32),
        K.astype(np.float32),
        D.astype(np.float32)
      )

      if not np.isnan(palito_2d[0,0]).any():
        px = int(palito_2d[0,0,0])
        py = int(palito_2d[0,0,1])
        cv2.circle(out, (px, py), 5, (0,0,255), -1)

    cv2.putText(out, f"point1: {point1}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(out, f"point2: {point2}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(out, f"distance: {distance}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    show_fitted(params.winname, out)

    k = cv2.waitKey(1) & 0xFF

    if k == ord('t'):
      point1 = point2
      point2 = punta_del_palito_en_coordenadas_de_camara
      distance = dist = np.linalg.norm(point1 - point2)
    elif k == ord('q'):
        break

  cap.release()


def main(mode):
  root = tk.Tk()
  params = SimpleNamespace(winname = "Webcam",
                marker_side_m = 0.045,
                screen_w = root.winfo_screenwidth(),
                screen_h = root.winfo_screenheight())
  root.destroy()

  cv2.namedWindow(params.winname, cv2.WINDOW_NORMAL)
  cv2.setWindowProperty(params.winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

  if mode=="photo":
    process_photo(params)
  elif mode=="video":
    process_video(params)

  cv2.destroyAllWindows()


main("video")
#main("photo")

