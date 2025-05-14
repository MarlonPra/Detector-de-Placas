import cv2
from inference_sdk import InferenceHTTPClient
import os
import time
import numpy as np  # Para procesamiento de color

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="nbzPzBGoE6rYTlQ9WAU9"
)

MODEL_ID = "reconocimiento_de_placas/1"  # Modelo de detección de placas

# --- OPCIÓN: Cambia esto para usar webcam o video ---
USE_VIDEO_FILE = True  # Cambia a True para usar plates.mp4
VIDEO_FILE_PATH = "plates.mp4"  # Nombre del archivo de video

if USE_VIDEO_FILE:
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    print(f"Usando archivo de video: {VIDEO_FILE_PATH}")
else:
    cap = cv2.VideoCapture(0)  # Cambia a 0 si usas otra cámara
    print("Usando webcam.")

frame_count = 0
output_dir = "placas_recortadas"
os.makedirs(output_dir, exist_ok=True)

last_detection_time = None  # Para controlar el tiempo desde la última placa detectada

print("Presiona 'q' para salir.")

# --- Paso 1: Detección local de recuadro amarillo tipo placa ---
def encontrar_recuadro_amarillo(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([18, 70, 70])
    upper_yellow = np.array([38, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mejor_rect = None
    mejor_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        # Solo considerar rectángulos con área y aspecto similar a una placa
        if area > 2000 and 2 < aspect_ratio < 6:
            if area > mejor_area:
                mejor_area = area
                mejor_rect = (x, y, w, h)
    return mejor_rect

capturando_placa_confirmada = False
inicio_captura_confirmada = 0
placa_confirmada_crop = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Preview - Detector de Placas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ahora = time.time()

    # --- Paso 1: Buscar recuadro amarillo tipo placa ---
    rect = encontrar_recuadro_amarillo(frame)

    # --- Paso 2: Captura intensiva de todos los recortes durante 1 segundo ---
    if capturando_placa_confirmada:
        # Guardar todos los recortes posibles durante 1 segundo
        if placa_confirmada_crop is not None:
            save_path = os.path.join(output_dir, f"placa_{frame_count}.jpg")
            cv2.imwrite(save_path, placa_confirmada_crop)
            print(f"[Captura intensiva] Placa recortada guardada: {save_path}")
            frame_count += 1
            last_detection_time = ahora
        # Finaliza la captura intensiva después de 1 segundo
        if ahora - inicio_captura_confirmada > 1:
            print("Fin de la captura intensiva de placa confirmada.")
            capturando_placa_confirmada = False
            inicio_captura_confirmada = 0
            placa_confirmada_crop = None
        # Saltar la verificación con Roboflow durante captura intensiva
        # (no ejecutar el resto del ciclo)
        pass
    elif rect is not None:
        x, y, w, h = rect
        print(f"[DEBUG] Recuadro amarillo detectado en local: x={x}, y={y}, w={w}, h={h}, aspecto={w/h:.2f}")
        temp_img = "frame_temp.jpg"
        cv2.imwrite(temp_img, frame)  # Enviar el frame completo
        print("[DEBUG] Enviando frame completo a Roboflow para confirmación...")
        result = CLIENT.infer(temp_img, model_id=MODEL_ID)
        print(f"[DEBUG] Respuesta de Roboflow: {result}")
        placa_detectada = False
        placa_coords = None
        for pred in result['predictions']:
            if pred.get('class', '') == 'license-plate' and pred['confidence'] > 0.5:
                placa_detectada = True
                placa_coords = pred
                print(f"[DEBUG] Roboflow CONFIRMA placa: {pred}")
                break
        if not placa_detectada:
            print("[DEBUG] Roboflow NO confirma que sea una placa.")
        if placa_detectada and placa_coords is not None:
            # Recortar la placa con margen extra
            x_center, y_center = int(placa_coords['x']), int(placa_coords['y'])
            w_lp, h_lp = int(placa_coords['width']), int(placa_coords['height'])
            # Margen del 10%
            margin_x = int(w_lp * 0.05)
            margin_y = int(h_lp * 0.05)
            x1 = max(0, x_center - w_lp // 2 - margin_x)
            y1 = max(0, y_center - h_lp // 2 - margin_y)
            x2 = min(frame.shape[1], x_center + w_lp // 2 + margin_x)
            y2 = min(frame.shape[0], y_center + h_lp // 2 + margin_y)
            placa_confirmada_crop = frame[y1:y2, x1:x2]
            print(f"[DEBUG] Recorte final de placa: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print("¡Placa CONFIRMADA por Roboflow! Iniciando captura intensiva por 2 segundos...")
            capturando_placa_confirmada = True
            inicio_captura_confirmada = ahora
    
    # --- Paso 4: Procesamiento igual que antes ---
    if frame_count > 0 and last_detection_time is not None:
        if time.time() - last_detection_time > 3:
            import subprocess
            import sys
            print(f"Han pasado más de 3 segundos sin nuevas placas. Procesando las {frame_count} placas recortadas...")
            subprocess.run([sys.executable, "leer_texto_placa.py"], check=True)
            from collections import Counter
            resultados_file = "resultados_texto_placas.txt"
            if os.path.exists(resultados_file):
                with open(resultados_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-frame_count:]  # Solo las últimas capturadas
                placas_detectadas = [line.strip().split(": ")[-1] for line in lines if ": " in line]
                # Excluir textos vacíos
                placas_validas = [p for p in placas_detectadas if 5 <= len(p.strip()) <= 6]
                if placas_validas:
                    placa_final = Counter(placas_validas).most_common(1)[0][0]
                    print(f"Placa más probable (moda entre {len(placas_validas)}): {placa_final}")
                else:
                    print("No se detectó texto válido en las placas recortadas (de 5 a 6 caracteres).")
            frame_count = 0
            last_detection_time = None  # Reiniciar el temporizador

cap.release()
cv2.destroyAllWindows()
if os.path.exists(temp_img):
    os.remove(temp_img)
print(f"Total de placas guardadas: {frame_count}")
