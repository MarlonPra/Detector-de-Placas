import cv2
from inference_sdk import InferenceHTTPClient
import os
import time  # <--- NUEVO: importar time

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="nbzPzBGoE6rYTlQ9WAU9"
)

MODEL_ID = "reconocimiento_de_placas/1"  # Modelo de detección de placas

cap = cv2.VideoCapture(1)  # Cambia a 0 si usas otra cámara
frame_count = 0
output_dir = "placas_recortadas"
os.makedirs(output_dir, exist_ok=True)

last_detection_time = None  # Para controlar el tiempo desde la última placa detectada

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mostrar preview de la cámara antes de la inferencia para fluidez
    cv2.imshow("Preview - Detector de Placas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    temp_img = "frame_temp.jpg"
    cv2.imwrite(temp_img, frame)
    result = CLIENT.infer(temp_img, model_id=MODEL_ID)

    best_conf = 0
    best_crop = None
    x1 = y1 = x2 = y2 = 0  # Inicializar para evitar errores si no hay predicción
    for pred in result['predictions']:
        if pred.get('class', '') == 'license-plate' and pred['confidence'] > best_conf:
            x1 = int(pred['x'] - pred['width'] / 2)
            y1 = int(pred['y'] - pred['height'] / 2)
            x2 = int(pred['x'] + pred['width'] / 2)
            y2 = int(pred['y'] + pred['height'] / 2)
            best_crop = frame[y1:y2, x1:x2]
            best_conf = pred['confidence']

    # Dibuja el rectángulo en el frame si hay una placa detectada
    if best_crop is not None and best_conf > 0:
        # Asegura que las coordenadas estén dentro del frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if best_crop is not None:
        save_path = os.path.join(output_dir, f"placa_{frame_count}.jpg")
        cv2.imwrite(save_path, best_crop)
        print(f"Placa recortada guardada: {save_path}")
        frame_count += 1
        last_detection_time = time.time()  # Actualizar el tiempo de la última detección

        # Cuando se hayan guardado 10 placas, ejecuta el reconocimiento y calcula la placa más frecuente
        if frame_count % 10 == 0:
            import subprocess
            import sys
            print("Procesando las 10 placas recortadas...")
            subprocess.run([sys.executable, "leer_texto_placa.py"], check=True)
            # Lee los resultados y calcula la placa más frecuente
            from collections import Counter
            resultados_file = "resultados_texto_placas.txt"
            if os.path.exists(resultados_file):
                with open(resultados_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-10:]  # Solo las últimas 10
                placas_detectadas = [line.strip().split(": ")[-1] for line in lines if ": " in line]
                if placas_detectadas:
                    placa_final = Counter(placas_detectadas).most_common(1)[0][0]
                    print(f"Placa más probable (moda entre las 10): {placa_final}")
                else:
                    print("No se detectaron placas válidas en las 10 imágenes.")
            frame_count = 0
            last_detection_time = None  # Reiniciar el temporizador

    # Si hay placas guardadas, han pasado 3 segundos desde la última detección y no se llegó a 10, procesa lo que hay
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
                if placas_detectadas:
                    placa_final = Counter(placas_detectadas).most_common(1)[0][0]
                    print(f"Placa más probable (moda entre las {frame_count}): {placa_final}")
            frame_count = 0
            last_detection_time = None  # Reiniciar el temporizador

cap.release()
cv2.destroyAllWindows()
if os.path.exists(temp_img):
    os.remove(temp_img)
print(f"Total de placas guardadas: {frame_count}")
