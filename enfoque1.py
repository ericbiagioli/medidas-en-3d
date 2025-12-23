from datetime import datetime
import time
import cv2
import numpy as np
import tkinter as tk
from types import SimpleNamespace
import pickle
import os

from helpers import show_fitted
from helpers import default_camera_matrix
from helpers import draw_results
from helpers import get_distance


def get_camera_camera_and_distortion_matrices(w, h):
  #if os.path.exists("calib-results.pkl"):
  #  with open("calib-results.pkl", "rb") as f:
  #    K, D = pickle.load(f)
  #  return K, D
  #else:
    return default_camera_matrix(w, h)



"""
Detecta marcadores ArUco y estima los vectores de rotación y traslación de cada
marcador detectado.

ASUMO QUE LOS ARUCOS SIENDO USADOS SON DEL DICCIONARIO 5X5_1000

Parámetros:

- img: imágen BGR (array numpy).
- marker_side_m: tamaño del lado del marcador, en metros.
- K: matriz de cámara. Es una matriz numpy de 3x3, calculada con
     default_camera_matrix o bien con algun proceso de calibración de cámara
     que permita mejores coeficientes.
- D: Coeficientes de distorsión. Es un array de 5x1, calculado con
     default_camera_matrix o con algun proceso de calibración de cámara que
     permita mejores coeficientes.

Valor de retorno:

Lista de objetos de la forma:
[{
    'id': int,
    'corners': [...],
    'rvec': <rvec>,
    `tvec`: <tvec>
 }, ...]

Donde
  id: es el valor numérico del ArUco,
  tvec: vector de translación (es la posición del centro del marcador en coordenadas de cámara).
  rvec: vector de rotación.

  Las coordenadas de cámara son así:
    Eje X de la cámara: hacia la derecha en la imagen.
    Eje Y de la cámara: hacia abajo en la imagen.
    Eje Z de la cámara: hacia adelante desde el foco (positivo hacia la escena).
"""
def detect_markers(img, marker_side_m: float, K, D, frame_prev=None):

    # ===== Detectar marcadores

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ===== Reducir flickering
    clahe = cv2.createCLAHE(
      clipLimit=2.0,
      tileGridSize=(8,8)
    )
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    ar_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    ar_pars = cv2.aruco.DetectorParameters()

    ar_pars.adaptiveThreshWinSizeMin = 5
    ar_pars.adaptiveThreshWinSizeMax = 23
    ar_pars.adaptiveThreshWinSizeStep = 4

    ar_pars.adaptiveThreshConstant = 7

    ar_pars.minCornerDistanceRate = 0.05
    ar_pars.minMarkerPerimeterRate = 0.03
    ar_pars.maxMarkerPerimeterRate = 4.0

    ar_pars.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    ar_pars.cornerRefinementWinSize = 7
    ar_pars.cornerRefinementMaxIterations = 100
    ar_pars.cornerRefinementMinAccuracy = 1e-6

    corners, ids, _ = cv2.aruco.detectMarkers(gray, ar_dict, parameters=ar_pars)

    if ids is None or corners is None or len(ids) == 0:
      return []

    # ===== Calcular tvec y rvec para cada marcador detectado

    # Puntos del marcador, en coordenadas del marcador
    half = marker_side_m / 2.0
    object_points = np.array([
      [-half,  half, 0],
      [ half,  half, 0],
      [ half, -half, 0],
      [-half, -half, 0]
    ], dtype=np.float32)

    ret = []
    for i, marker_id in enumerate(ids.flatten()):
      image_points = (corners[i])[0].astype(np.float32)

      ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, D, cv2.SOLVEPNP_IPPE_SQUARE)
      cv2.solvePnPRefineLM(object_points, image_points, K, D, rvec, tvec)

      ret.append({'id': int(marker_id),
                  'corners': corners[i].reshape(-1, 2).tolist(),
                  'rvec':rvec,
                  'tvec':tvec})

    return ret




def to_cammera_coords(obj_coords, rvec, tvec):
  # cv2.Rodrigues transforma una representacion vectorial de una rotacion en
  # la representación matricial, que llamaremos R.
  R, _ = cv2.Rodrigues(rvec)
  return R @ obj_coords.reshape(3,1) + tvec




class Writer:
  def __init__(self, img):
    self.cx = 50
    self.cy = 50
    self.img = img

  def write(self, text):
    cv2.putText(self.img,
      text, (self.cx, self.cy),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (0,255,0),
      2)
    self.cy = self.cy + 30





def process_video(params):



  palito_a = None
  palito_b = None
  distance = get_distance(palito_a, palito_b)
  # palito_aruco es la punta del palito en coordenadas aruco
  palito_aruco = {}
  #palito_aruco[63] = np.array([-0.041, -0.150, -0.0025])
  palito_aruco[63] = np.array([0.0, 0.0, 0.0])
  palito_aruco[63] = palito_aruco[63].astype(np.float64)




  cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)

  # IP WebCam pro
  #cap = cv2.VideoCapture("http://127.0.0.1:8080/video")
  if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

  cam_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  cam_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  print(cam_w, cam_h)




  K, D = get_camera_camera_and_distortion_matrices(cam_w, cam_h)




  while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame de la cámara")
        exit(0)

    ## use a fixed photo
    #ret = True
    #frame = cv2.imread("photo.jpg",  cv2.IMREAD_COLOR)


    markers = detect_markers(frame, params.marker_side_m, K, D)


    # Calcular el puntito aruco como el promedio de todos los puntitos arucos
    # correspondientes a cada uno de los markers detectados
    # Sistema de coordenadas del marcador:
    # El marcador tiene origen en su centro.
    # +x: Apunta hacia la derecha en la imagen del marcador.
    # +y: Apunta hacia abajo en la imagen del marcador.
    # +z: Apunta perpendicular al plano del marcador, hacia la cámara cuando
    #     el marcador está frente a la cámara.

    ## USO SOLO EL MARKER 63

    rvec = None
    tvec = None
    mark = None
    palito_camara = None

    for marker in markers:
      if marker['id'] != 63:
        continue
      mark = marker
      break

    if mark is not None:
      rvec = mark['rvec'].astype(np.float64)
      tvec = mark['tvec'].reshape(3, 1).astype(np.float64)

      # palito_camera es la punta del palito en coordenadas de camara
      palito_camara = to_cammera_coords(palito_aruco[63], rvec, tvec)
      palito_camara = palito_camara.astype(np.float64).reshape(3,1)

    ## Dibujar contornos
    out = draw_results(frame, markers, K, D, params.marker_side_m)

    ## Dibujar la punta del palito
    if palito_camara is not None:
      palito_2d, _ = cv2.projectPoints(palito_camara, rvec,
        tvec.astype(np.float64), K.astype(np.float64), D.astype(np.float64))

      print(palito_2d)
      ih, iw = out.shape[:2]
      if not np.isnan(palito_2d[0,0]).any():
        px = int(palito_2d[0,0,0])
        py = int(palito_2d[0,0,1])
        if 0 <= px < iw and 0 <= py < ih:
            cv2.circle(out, (px, py), 5, (0, 0, 255), -1)





    ## Escribir textos
    distance = get_distance(palito_a, palito_b)


    w = Writer(out)
    w.write(f"palito_a = {palito_a}")
    w.write(f"palito_b = {palito_b}")
    w.write(f"dist: {distance}")

    show_fitted(params.winname, out)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('a'):
      palito_a = palito_camara
    elif k == ord('b'):
      palito_b = palito_camara
    elif k == ord('q'):
        break

  cap.release()


def main():
  root = tk.Tk()
  params = SimpleNamespace(winname = "Webcam", marker_side_m = 0.043,
                screen_w = root.winfo_screenwidth(),
                screen_h = root.winfo_screenheight())
  root.destroy()
  cv2.namedWindow(params.winname, cv2.WINDOW_NORMAL)
  cv2.setWindowProperty(params.winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
  process_video(params)
  cv2.destroyAllWindows()


main()

