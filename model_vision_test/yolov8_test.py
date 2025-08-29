import cv2
from ultralytics import YOLO
import time

# Cargar el modelo (usa 'yolov8n.pt' para pruebas rápidas)
model = YOLO('./models/yolov8n.pt')   # Puedes cambiar a 'yolov8s.pt', 'yolov8m.pt', etc.

# Abrir cámara (índice 0 es la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Configurar resolución (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables para calcular FPS
prev_frame_time = 0
curr_frame_time = 0

print("Presiona 'q' para salir...")

while True:
    # Leer frame
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    # Hacer inferencia
    results = model(frame, verbose=False)  # verbose=False para no saturar la consola
    class_names = model.names
    print(results)
    # Dibujar detecciones en el frame
    annotated_frame = results[0].plot()  # Esto incluye cajas, etiquetas y confianza

    # Calcular FPS
    curr_frame_time = time.time()
    fps = 1 / (curr_frame_time - prev_frame_time)
    prev_frame_time = curr_frame_time

    # Mostrar FPS en pantalla
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el video
    cv2.imshow("YOLOv8 - Webcam", annotated_frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()