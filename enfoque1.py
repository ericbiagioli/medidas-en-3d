from datetime import datetime
import time
import cv2
import numpy as np
import tkinter as tk
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
def detect_markers(img, marker_side_m: float, K, D):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret = []

    ar_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    ar_pars = cv2.aruco.DetectorParameters()

    corners_list, ids, _ = cv2.aruco.detectMarkers(gray, ar_dict, parameters=ar_pars)

    if ids is None or corners_list is None or len(ids) == 0:
      return ret

    # Puntos del marcador, en coordenadas del marcador
    half = marker_side_m / 2.0
    object_points = np.array([
      [-half,  half, 0],
      [ half,  half, 0],
      [ half, -half, 0],
      [-half, -half, 0]
    ], dtype=np.float32)

    for i, marker_id in enumerate(ids.flatten()):
      image_points = (corners_list[i])[0].astype(np.float32)

      ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
      ret.append({'id': int(marker_id),
                  'corners': corners_list[i].reshape(-1, 2).tolist(),
                  'rvec':rvec,
                  'tvec':tvec})

    return ret


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

class PointCaptureControl:
  def __init__(self, name):
    self.name = name
    self.point = None
    self.capturing = False
    self.captured = []
    self.starttime = None

  def start(self):
    self.point = None
    self.capturing = True
    self.captured = []
    self.starttime = datetime.now()

  def process(self, p):
    if self.capturing == True:
      if(datetime.now() - self.starttime).total_seconds() < 3.0:
        self.captured.append(p)
      else:
        self.capturing = False
        if len(self.captured) == 0:
          self.point = None
        else:
          arr = np.array(self.captured).reshape(len(self.captured), 3)
          self.point = np.mean(arr, axis=0).reshape(3,1)
          self.captured = []
          self.starttime = None

status = dict(m=PointCaptureControl('m'), n=PointCaptureControl('n'))

def get_distance(p1, p2):
  if p1 is None or p2 is None:
    return None
  return np.linalg.norm(p1- p2)


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

  distance = None

  while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer el frame de la cámara")
        exit(0)

    markers = detect_markers(frame, params.marker_side_m, K, D)

    for idx, marker in enumerate(markers):
      punta_del_palito_en_coordenadas_de_camara = None

      # Sistema de coordenadas del marcador: El marcador tiene origen en su centro.
      # +x: Apunta hacia la derecha en la imagen del marcador.
      # +y: Apunta hacia abajo en la imagen del marcador.
      # +z: Apunta perpendicular al plano del marcador, hacia la cámara cuando el marcador está frente a la cámara.
      rvec = marker['rvec']
      tvec = marker['tvec'].reshape(3, 1)

      # cv2.Rodrigues transforma una representacion vectorial de una rotacion en
      # la representación matricial, que llamaremos R. La necesitaremos para
      # calcular donde está la punta del palito.
      R, _ = cv2.Rodrigues(rvec)

      # Definir las posición de la punta del palito en coordenadas del marcador.
      punta_del_palito_en_coordenadas_aruco = np.array([0.0, -0.27, 0.0])
      punta_del_palito_en_coordenadas_de_camara = R @ punta_del_palito_en_coordenadas_aruco.reshape(3,1) + tvec
      centro_del_aruco = tvec

      status['m'].process(punta_del_palito_en_coordenadas_de_camara)
      status['n'].process(punta_del_palito_en_coordenadas_de_camara)

    distance=get_distance(status['m'].point, status['n'].point)

    ############################################################################
    # Dibujar los marcadores
    out = draw_results(frame, markers, K, D, params.marker_side_m)

    # Escribir textos que ayudan a debuguear
    for idx, marker in enumerate(markers):
      cv2.putText(out, f"Aruco: {centro_del_aruco}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
      cv2.putText(out, f"Palito: {punta_del_palito_en_coordenadas_de_camara}",   (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

      ## Dibujar la punta del palito
      p_mark = punta_del_palito_en_coordenadas_aruco.astype(np.float32).reshape(1,1,3)
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


    cv2.putText(out, f"point m: {status['m'].point}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(out, f"point n: {status['n'].point}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(out, f"distance: {distance}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    show_fitted(params.winname, out)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
      status['m'].start()
    elif k == ord('n'):
      status['n'].start()
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

