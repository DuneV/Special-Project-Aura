import cv2
import numpy as np
import time

# Importa FastSAM desde la librería ultralytics
from ultralytics import FastSAM

# Cargar el modelo FastSAM
# Se recomienda la versión 's' (small) para video en tiempo real debido a su velocidad.
# El modelo se descargará automáticamente la primera vez que se ejecute.
model = FastSAM("FastSAM-s.pt")

# Abre la cámara (índice 0 es la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Configurar resolución (opcional, para un mejor rendimiento)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables para calcular el FPS
prev_frame_time = 0
curr_frame_time = 0

print("Presiona 'q' para salir del video...")

while True:
    # Leer un fotograma de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo acceder a la cámara o leer el fotograma.")
        break

    # --- Realizar la inferencia con FastSAM ---
    # En este caso, usamos el modo de segmentar "todo" en el fotograma.
    # Los parámetros de configuración son cruciales para el rendimiento.
    results = model(frame, 
                    device="cpu", # Usa CPU para compatibilidad, o '0' para GPU
                    retina_masks=True, 
                    imgsz=640,  # Asegúrate de que este valor coincida con la resolución de tu cámara
                    conf=0.4, 
                    iou=0.9)

    # --- Visualización de los resultados ---
    # El método .plot() de YOLOv8 no existe en FastSAM de la misma manera,
    # así que lo implementamos manualmente aquí.

    # Convertir el fotograma a un formato editable si es necesario.
    annotated_frame = frame.copy()

    if results is not None and len(results) > 0 and results[0].masks:
        # Obtener las máscaras de segmentación.
        masks = results[0].masks.data.cpu().numpy()
        
        # Crear una imagen en blanco para dibujar las máscaras.
        mask_overlay = np.zeros_like(annotated_frame)
        
        # Iterar sobre cada máscara y superponerla en el fotograma.
        for mask in masks:
            # Redimensionar la máscara para que coincida con el tamaño del fotograma
            mask_resized = cv2.resize(mask, 
                                      (annotated_frame.shape[1], annotated_frame.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask_resized > 0.5
            
            # Asignar un color aleatorio a cada máscara para diferenciar objetos
            color = np.random.randint(0, 255, size=(3,))
            mask_overlay[mask_bool] = color
        
        # Combinar el fotograma original con la superposición de máscaras
        alpha = 0.5  # Transparencia
        annotated_frame = cv2.addWeighted(annotated_frame, 1 - alpha, mask_overlay, alpha, 0)

    # --- Calcular y mostrar el FPS ---
    curr_frame_time = time.time()
    fps = 1 / (curr_frame_time - prev_frame_time)
    prev_frame_time = curr_frame_time

    # Mostrar el FPS en el fotograma
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el video resultante en una ventana
    cv2.imshow("FastSAM - Webcam", annotated_frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
