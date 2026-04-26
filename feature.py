# pylint: disable=no-member,no-name-in-module
from ultralytics import YOLO
import cv2
import numpy as np

# ── Configuración ──────────────────────────────────────────────
FIRE_HSV_LOW  = np.array([0,  120, 120])
FIRE_HSV_HIGH = np.array([35, 255, 255])
MIN_FIRE_AREA = 500          # px² mínimo para considerar detección válida
CROSSHAIR_SIZE = 16          # tamaño de la cruz del punto crítico

model = YOLO("weights/best.pt")

cv2.namedWindow("Fire Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Fire Detection", 800, 600)

media = cv2.VideoCapture("test_videos/fire.mp4")
if not media.isOpened():
    print("Error abriendo el video")
else:
    print("Video abierto correctamente")


def get_critical_point(mask: np.ndarray, offset_x: int, offset_y: int):
    """
    Calcula el punto más crítico del fuego usando centroide ponderado
    por intensidad de la máscara. El píxel más brillante (más blanco
    en la máscara) tiene mayor peso → apunta al núcleo más caliente.

    Retorna: (cx, cy, area, radio_accion) o None si no hay fuego suficiente.
    """
    area = cv2.countNonZero(mask)
    if area < MIN_FIRE_AREA:
        return None

    # --- Centroide ponderado por intensidad ---
    # Cada píxel de la máscara tiene valor 0-255.
    # np.argwhere devuelve las coordenadas de píxeles activos.
    coords = np.column_stack(np.where(mask > 0))   # shape (N, 2): [row, col]
    weights = mask[mask > 0].astype(np.float64)    # intensidad de cada píxel

    total_weight = weights.sum()
    if total_weight == 0:
        return None

    # weighted mean: fila=y, col=x  (recuerda que np.where devuelve [row, col])
    cy_local = (coords[:, 0] * weights).sum() / total_weight
    cx_local = (coords[:, 1] * weights).sum() / total_weight

    # Convertir a coordenadas del frame completo
    cx = int(offset_x + cx_local)
    cy = int(offset_y + cy_local)

    # Radio de acción estimado (raíz del área → círculo equivalente)
    radio = int(np.sqrt(area / np.pi))

    return cx, cy, area, radio


def draw_target(frame: np.ndarray, cx: int, cy: int, radio: int, area: int):
    """Dibuja la visualización del punto crítico sobre el frame."""

    # Círculo exterior: zona de acción recomendada
    cv2.circle(frame, (cx, cy), radio,      (0, 255, 255), 2)   # amarillo
    cv2.circle(frame, (cx, cy), radio // 2, (0, 165, 255), 1)   # naranja (núcleo)

    # Cruz / crosshair en el punto exacto
    cv2.line(frame,
             (cx - CROSSHAIR_SIZE, cy), (cx + CROSSHAIR_SIZE, cy),
             (0, 0, 255), 2)
    cv2.line(frame,
             (cx, cy - CROSSHAIR_SIZE), (cx, cy + CROSSHAIR_SIZE),
             (0, 0, 255), 2)

    # Punto central relleno
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Texto informativo
    label = f"TARGET ({cx},{cy}) | area={area}px2 | r={radio}px"
    cv2.putText(frame, label,
                (cx + CROSSHAIR_SIZE + 4, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA)


# ── Loop principal ─────────────────────────────────────────────
ret, frame = media.read()   # descarta el primer frame (a veces incompleto)

while True:
    ret, frame = media.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Seguridad: recorte dentro de límites del frame
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Máscara HSV del fuego dentro del bounding box
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, FIRE_HSV_LOW, FIRE_HSV_HIGH)

        # Suavizado morfológico para eliminar ruido pequeño
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        result = get_critical_point(mask, offset_x=x1, offset_y=y1)
        if result is None:
            continue

        cx, cy, area, radio = result
        draw_target(annotated, cx, cy, radio, area)
        print(f" Punto crítico: ({cx},{cy}) | Área: {area}px² | Radio acción: {radio}px")

    cv2.imshow("Fire Detection", annotated)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

media.release()
cv2.destroyAllWindows()