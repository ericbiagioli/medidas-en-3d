from datetime import datetime
import time
import cv2
import numpy as np
import tkinter as tk
from types import SimpleNamespace

from helpers import show_fitted
from helpers import default_camera_matrix
from helpers import draw_results
from helpers import get_distance


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

      ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
      cv2.solvePnPRefineLM(object_points, image_points, K, D, rvec, tvec)

      ret.append({'id': int(marker_id),
                  'corners': corners[i].reshape(-1, 2).tolist(),
                  'rvec':rvec,
                  'tvec':tvec})

    return ret



"""
Procesa frame por frame.

En cada frame, chequea ArUcos existentes y envía los arucos existentes para ser
procesados en el objeto de estado. Esta lista de ArUcos detectados podría estar
vacía.

Una vez procesados los (potencialmente 0) arucos detectados, se procede a
escribir los textos y demás cosas que corresponda escribir arriba del frame, y
a dibujarlo.


Posteriormente, se procede a leer el teclado y enviar la tecla leída al objeto
de estado.
"""

class LastNSeconds:
  def __init__(self):
    self.captured = []
    self.TIME_INTERVAL = 20.0

  def add(self, p):
    t = datetime.now()
    self.captured.append( (t, p))
    i = 0
    while i < len(self.captured) and (t - self.captured[i][0]).total_seconds() > self.TIME_INTERVAL:
      i += 1
    self.captured = self.captured[i:]
    #self.filter_outliers()


  def filter_outliers(self):
    points = np.array([p for (_, p) in self.captured])
    centroid = points.mean(axis = 0)
    std = points.std(axis = 0)
    dists = np.linalg.norm(points - centroid, axis=1)
    mu = dists.mean()
    sigma = dists.std()

    if sigma == 0:
        mask = np.ones(len(points), dtype=bool)
    else:
        mask = (dists >= mu - 1.0 * sigma) & (dists <= mu + 1.0 * sigma)

    self.captured = points[mask]


  def ncaptures(self):
    return len(self.captured)

  def stats(self):
    points = np.array([p for (_, p) in self.captured])
    mean = points.mean(axis = 0)
    std = points.std(axis = 0)
    return mean, std

  def ok(self):
    points = np.array([p for _, p in self.captured])
    mean = points.mean(axis = 0)
    std = points.std(axis = 0)
    ok1 = (len(self.captured) > 9)
    ok2 = (std < 0.001).all()
    return (ok1 and ok2), mean, std



def obj_to_cammera_coords(obj_coords, rvec, tvec):
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
    cv2.putText(self.img, text, (self.cx, self.cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    self.cy = self.cy + 30

def process_video(params):
  device = 1
  cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

  stats : Dict[int, LastNSeconds] = {}

  point_m = None
  point_n = None
  distance = get_distance(point_m, point_n)
  capturing_m = False
  capturing_n = False


  if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

  ret, frame = cap.read()
  if not ret:
    print("No se pudo leer un frame de la cámara")
    exit()

  h, w = frame.shape[:2]
  K, D = default_camera_matrix(w, h)

  while True:
    ret, frame = cap.read()

    ## use a frame
    #ret = True
    #frame = cv2.imread("photo.jpg",  cv2.IMREAD_COLOR)
    #print("reading")

    if not ret:
        print("No se pudo leer el frame de la cámara")
        exit(0)

    markers = detect_markers(frame, params.marker_side_m, K, D)


    for marker in markers:
      if marker['id'] not in stats:
        stats[marker['id']] = LastNSeconds()

      # Sistema de coordenadas del marcador:
      # El marcador tiene origen en su centro.
      # +x: Apunta hacia la derecha en la imagen del marcador.
      # +y: Apunta hacia abajo en la imagen del marcador.
      # +z: Apunta perpendicular al plano del marcador, hacia la cámara cuando
      #     el marcador está frente a la cámara.
      rvec = marker['rvec']
      tvec = marker['tvec'].reshape(3, 1)

      # palito_aruco es la punta del palito en coordenadas aruco
      # palito_camera es la punta del palito en coordenadas de camara
      palito_aruco = np.array([0.0, -0.27, 0.0])
      palito_camara = obj_to_cammera_coords(palito_aruco, rvec, tvec)
      stats[marker['id']].add(palito_camara)

    out = draw_results(frame, markers, K, D, params.marker_side_m)
    distance = get_distance(point_m, point_n)

    # Escribir textos
    w = Writer(out)
    for marker in markers:
      m_id = marker['id']
      rvec = marker['rvec']
      tvec = marker['tvec'].reshape(3, 1)
      palito_aruco = np.array([0.0, -0.27, 0.0])
      palito_camara = obj_to_cammera_coords(palito_aruco, rvec, tvec)

      ncaptures = stats[m_id].ncaptures()
      avg,std = stats[m_id].stats()
      w.write(f"Aruco instantaneo [{marker['id']}] = {tvec}")
      w.write(f"Palito instantaneo [{marker['id']}] = {palito_camara}")
      w.write(f"ncaptures[{marker['id']}] = {ncaptures}")
      w.write(f"avg[{marker['id']}] = {avg}")
      w.write(f"std[{marker['id']}] = {std}")

      ## Dibujar la punta del palito
      p_mark = palito_aruco.astype(np.float32).reshape(1,1,3)
      palito_2d, _ = cv2.projectPoints( p_mark, rvec.astype(np.float32),
        tvec.astype(np.float32), K.astype(np.float32), D.astype(np.float32))
      if not np.isnan(palito_2d[0,0]).any():
        px = int(palito_2d[0,0,0])
        py = int(palito_2d[0,0,1])
        cv2.circle(out, (px, py), 5, (0,0,255), -1)

    ok = False
    if 63 in stats.keys():
      ok, centroid, deviation = stats[63].ok()

    if capturing_m:
      if ok:
        capturing_m = False
        point_m = centroid
      else:
        point_m = "capturing"

    if capturing_n:
      if ok:
        capturing_n = False
        point_n = centroid
      else:
        point_n = "capturing"

    distance = get_distance(point_m, point_n)

    w.write(f"m = {point_m}")
    w.write(f"n = {point_n}")
    w.write(f"dist: {distance}")

    show_fitted(params.winname, out)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
      capturing_m = True
    elif k == ord('n'):
      capturing_n = True
    elif k == ord('q'):
        break

  cap.release()


def main():
  root = tk.Tk()
  params = SimpleNamespace(winname = "Webcam", marker_side_m = 0.045,
                screen_w = root.winfo_screenwidth(),
                screen_h = root.winfo_screenheight())
  root.destroy()
  cv2.namedWindow(params.winname, cv2.WINDOW_NORMAL)
  cv2.setWindowProperty(params.winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
  process_video(params)
  cv2.destroyAllWindows()


main()

