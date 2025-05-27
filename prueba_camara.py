import cv2
import threading

frame = None
running = True

def capture_thread():
    global frame, running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        running = False
        return
    while running:
        ret, img = cap.read()
        if ret:
            frame = img
    cap.release()

t = threading.Thread(target=capture_thread)
t.start()

cv2.namedWindow("Camara Fluida", cv2.WINDOW_NORMAL)

try:
    while running:
        if frame is not None:
            cv2.imshow("Camara Fluida", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            running = False
            break
except KeyboardInterrupt:
    running = False

t.join()
cv2.destroyAllWindows()
