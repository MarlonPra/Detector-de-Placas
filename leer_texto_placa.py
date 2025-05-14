import glob
from inference_sdk import InferenceHTTPClient
import cv2
import os

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="nbzPzBGoE6rYTlQ9WAU9"
)

MODEL_ID = "license-ocr-qqq6v/3"  # Modelo de reconocimiento de caracteres

RESULTS_FILE = "resultados_texto_placas.txt"
output_dir = "placas_recortadas"

# Procesa todas las imágenes de placas recortadas
placas = sorted(glob.glob(os.path.join(output_dir, "placa_*.jpg")))

for img_path in placas:
    result = CLIENT.infer(img_path, model_id=MODEL_ID)
    print(f"[DEBUG] OCR respuesta cruda Roboflow para {os.path.basename(img_path)}: {result}")
    # Extrae y ordena los caracteres (sin eliminar duplicados ni agrupar)
    detected_chars = []
    for prediction in result['predictions']:
        x1 = int(prediction['x'] - prediction['width'] / 2)
        label = prediction['class']
        if label.isalnum():
            detected_chars.append((x1, label))
    print(f"[DEBUG] Caracteres detectados (sin ordenar): {detected_chars}")
    detected_chars.sort()
    print(f"[DEBUG] Caracteres detectados (ordenados): {detected_chars}")
    # Tomar todos los caracteres en orden, sin eliminar duplicados
    plate_text = ''.join([c for _, c in detected_chars])
    print(f"[DEBUG] Texto final de placa: {plate_text}")
    print(f"{os.path.basename(img_path)}: {plate_text}")
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{os.path.basename(img_path)}: {plate_text}\n")
print(f"Resultados guardados en {RESULTS_FILE}")

# Eliminar las imágenes procesadas
import glob
for img_path in glob.glob(os.path.join(output_dir, "placa_*.jpg")):
    os.remove(img_path)
print("Imágenes recortadas eliminadas.")
