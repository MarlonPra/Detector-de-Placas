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
    # Extrae y ordena los caracteres
    detected_chars = []
    for prediction in result['predictions']:
        x1 = int(prediction['x'] - prediction['width'] / 2)
        label = prediction['class']
        if label.isalnum():
            detected_chars.append((x1, label))
    detected_chars.sort()
    # Elimina duplicados consecutivos y agrupa caracteres cercanos
    min_dist = 15
    grouped_chars = []
    for x, char in detected_chars:
        if not grouped_chars or abs(x - grouped_chars[-1][0]) > min_dist:
            grouped_chars.append((x, char))
    plate_no_dupes = []
    for _, c in grouped_chars:
        if not plate_no_dupes or c != plate_no_dupes[-1]:
            plate_no_dupes.append(c)
    plate_text = ''.join(plate_no_dupes[:6])
    print(f"{os.path.basename(img_path)}: {plate_text}")
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{os.path.basename(img_path)}: {plate_text}\n")
print(f"Resultados guardados en {RESULTS_FILE}")

# Eliminar las imágenes procesadas
import glob
for img_path in glob.glob(os.path.join(output_dir, "placa_*.jpg")):
    os.remove(img_path)
print("Imágenes recortadas eliminadas.")
