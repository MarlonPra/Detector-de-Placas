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
- Python 3.11 o inferior
- Paquetes requeridos:
  - inference-sdk
  - opencv-python

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
1. Ejecutar el script de detección:
```bash
python detectar_placa_y_guardar.py
```

2. El programa iniciará la cámara y comenzará a detectar placas. Presiona 'q' para salir.

3. Una vez que se hayan detectado 10 placas, el sistema automáticamente:
   - Recortará las placas detectadas
   - Procesará el texto de las placas
   - Guardará los resultados en `resultados_texto_placas.txt`
   - Eliminará las imágenes procesadas

## Notas
- El sistema utiliza modelos de Roboflow para la detección de placas y el reconocimiento de texto
- Se requiere una cámara web conectada al sistema
- Los resultados se guardan en el directorio `placas_recortadas` y en el archivo `resultados_texto_placas.txt`

## Estructura del Proyecto
- `detectar_placa_y_guardar.py`: Script principal para la detección de placas
- `leer_texto_placa.py`: Script para el reconocimiento de texto de las placas
- `requirements.txt`: Lista de dependencias del proyecto
- `placas_recortadas/`: Directorio donde se guardan las placas recortadas
- `resultados_texto_placas.txt`: Archivo donde se guardan los resultados del OCR

## Licencia
Este proyecto está bajo la licencia MIT.