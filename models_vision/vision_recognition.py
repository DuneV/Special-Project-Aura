import cv2
from ultralytics import YOLO
import time
import os

# ==============================
# CLASE de video recognition
# ==============================


class VideoRecognition:
    
    def __init__(self, input_video_path: str, output_video_path: str, model: str):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.model_name = model
        self.model = YOLO(self.model_name)
        self.cap = None
        self.out = None
        self.prev_frame_time = 0
        self.frame_count = 0

    def video_analyzer(self):

        if not os.path.exists(self.input_video_path):
            raise FileNotFoundError(f"Video no encontrado: {self.input_video_path}")        
        self.cap = cv2.VideoCapture(self.input_video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def writer_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Formato MP4
        self.out = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )

    def runtime(self):

        try:
            self.video_analyzer()
            self.writer_video()
            while True:
                self.ret, self.frame = self.cap.read()
                if not self.ret:
                    print("Fin del video o error al leer frame. -Q.Q-")
                    break

                self.frame_count += 1
                results = self.model(self.frame, verbose=False)
                annotated_frame = results[0].plot() 

                # Cálculo de FPS
                curr_time = time.time()
                if self.prev_frame_time > 0:
                    fps_display = 1 / (curr_time - self.prev_frame_time)
                else:
                    fps_display = 0
                self.prev_frame_time = curr_time

                # Mostrar FPS en el video
                cv2.putText(annotated_frame, f"FPS: {fps_display:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Guardar frame procesado
                self.out.write(annotated_frame)
        except Exception as e:
            print(f"Error durante el procesamiento: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
