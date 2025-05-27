# Detector de Placas

## Descripción
Este proyecto es un sistema de detección y reconocimiento de placas de vehículos utilizando inteligencia artificial. El sistema utiliza una cámara web para detectar placas en tiempo real y extraer el texto de las placas detectadas.

## Participantes
- Marlon Pastrana Moreno
- Ana Victoria Diaz Marquinez
- Camilo Andres Gomez Morales
- Julian Andres Roldan Gaita
- Kevin Santiago Garzon Mazabel
- Brayan Pantoja
- Jordy Vargas
- Julian Hoyos

## Funcionalidades
- Detección de placas en tiempo real usando una cámara web
- Recorte automático de las placas detectadas
- Reconocimiento de texto de las placas usando OCR
- Almacenamiento de los resultados en un archivo de texto
- Eliminación automática de imágenes procesadas

## Requisitos
- Python 3.12 o inferior (según las dependencias actuales)
- Paquetes requeridos:
  - ultralytics==8.3.144
  - opencv-python==4.11.0.86
  - torch==2.7.0
  - torchvision==0.22.0
  - numpy==2.2.6
  - pillow (opcional)
  - matplotlib (opcional)
  - pandas (opcional)

## Instalación
1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd Detector-de-Placas
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Uso
1. Ejecutar el script principal de detección:
```bash
python placa_detect.py
```

2. El programa iniciará la cámara y comenzará a detectar placas. Presiona 'q' o 'ESC' para salir.

3. El sistema detectará placas en tiempo real, recortará las regiones detectadas y procesará el texto de las placas. Los resultados se mostrarán en pantalla y se podrán almacenar según la lógica del script.

## Notas
- El sistema utiliza modelos previamente entrenados (en formato .pt) para la detección y reconocimiento de placas
- Se requiere una cámara web conectada al sistema
- Los resultados pueden guardarse en archivos de texto o imágenes procesadas según la configuración del script

## Estructura del Proyecto
- `placa_detect.py`: Script principal para la detección y reconocimiento de placas
- `prueba_camara.py`: Script de prueba para la cámara web
- `best.pt`, `detector_placa.pt`, `license_plate_detector.pt`: Modelos de IA utilizados
- `plates.mp4`: Video de ejemplo para pruebas
- `requirements.txt`: Lista de dependencias del proyecto
- `leer_texto_placa.py`: Script para el reconocimiento de texto de las placas
- `placas_recortadas/`: Directorio donde se guardan las placas recortadas
- `resultados_texto_placas.txt`: Archivo donde se guardan los resultados del OCR

## Licencia
Este proyecto está bajo la licencia MIT.