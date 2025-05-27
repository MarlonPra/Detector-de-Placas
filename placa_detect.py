import cv2
import threading
import time
from ultralytics import YOLO
from collections import Counter

MODEL_OCR_PATH = 'best.pt'
MODEL_PLATE_PATH = 'license_plate_detector.pt'

ocr_model = YOLO(MODEL_OCR_PATH)
plate_detector = YOLO(MODEL_PLATE_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise FileNotFoundError('No se pudo abrir la camara.')

WINDOW_NAME = "Deteccion de placa"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

frame = None
running = True
placas_detectadas = []
lock = threading.Lock()

# Variables compartidas para dibujar resultados
last_boxes = []
last_texts = []

def captura():
    global frame, running
    while running:
        ret, img = cap.read()
        if ret:
            with lock:
                frame = img.copy()
        time.sleep(0.01)

def deteccion():
    global frame, running, placas_detectadas, last_boxes, last_texts
    while running:
        with lock:
            if frame is None:
                continue
            img = frame.copy()
        frame_resized = cv2.resize(img, (320, 240))
        plates = plate_detector(frame_resized)[0]
        scale_x = img.shape[1] / frame_resized.shape[1]
        scale_y = img.shape[0] / frame_resized.shape[0]
        boxes_to_draw = []
        texts_to_draw = []
        for plate_box in plates.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, plate_box)
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue
            results_ocr = ocr_model(plate_crop)
            boxes = results_ocr[0].boxes.xyxy.cpu().numpy()
            classes = results_ocr[0].boxes.cls.cpu().numpy()
            class_names = ocr_model.names
            outputs = []
            for box, cls in zip(boxes, classes):
                outputs.append((box, int(cls)))
            outputs.sort(key=lambda x: x[0][0])
            filtered = outputs[:6]
            placa_texto = ''.join([class_names[cls] for _, cls in filtered])
            placa_texto = placa_texto.strip().replace(' ', '')
            print(f"Placa detectada: '{placa_texto}' (len={len(placa_texto)})")
            if len(placa_texto) >= 4:
                placas_detectadas.append(placa_texto)
            boxes_to_draw.append((x1, y1, x2, y2))
            texts_to_draw.append((placa_texto, x1, y1))
        with lock:
            last_boxes = boxes_to_draw
            last_texts = texts_to_draw
        time.sleep(1.0)  # Ajusta la frecuencia de inferencia

t_cap = threading.Thread(target=captura)
t_det = threading.Thread(target=deteccion)
t_cap.start()
t_det.start()

try:
    while running:
        with lock:
            if frame is not None:
                frame_to_show = frame.copy()
                # Dibuja los resultados detectados por el hilo de inferencia
                for (x1, y1, x2, y2) in last_boxes:
                    cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
                for (text, x1, y1) in last_texts:
                    cv2.putText(frame_to_show, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow(WINDOW_NAME, frame_to_show)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            running = False
            break
except KeyboardInterrupt:
    running = False

t_cap.join()
t_det.join()
cap.release()
cv2.destroyAllWindows()

def placa_mas_probable(placas):
    if not placas:
        print("No se detecto ninguna placa valida.")
        return
    conteo = Counter(placas)
    placa, repeticiones = conteo.most_common(1)[0]
    print(f'Placa mas probable: {placa} (detectada {repeticiones} veces)')
    print(f'Todas las placas detectadas (conteo): {conteo}')

print(f"Cantidad de placas detectadas: {len(placas_detectadas)}")
print(f"Primeras placas detectadas: {placas_detectadas[:5]}")
placa_mas_probable(placas_detectadas)
