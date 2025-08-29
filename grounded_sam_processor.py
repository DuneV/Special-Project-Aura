# grounded_sam_processor.py
import cv2
import numpy as np
from ultralytics import SAM

class UltralyticsSAMProcessor:
    def __init__(self, model_path="sam2.1_b.pt"):
        """
        Inicializar con modelo SAM/SAM2 de Ultralytics
        
        Modelos disponibles:
        - "sam2.1_b.pt" (pequeño y rápido)
        - "sam2.1_l.pt" (grande y preciso) 
        - "sam.pt" (SAM original)
        """
        try:
            self.model = SAM(model_path)
            print(f"Loaded Ultralytics SAM model: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure to download the model first:")
            print(f"from ultralytics import SAM; SAM('{model_path}')")
            raise
    
    def process_frame(self, frame, points=None, boxes=None, labels=None):
        """
        Procesar frame con SAM/SAM2 de Ultralytics
        
        Args:
            frame: imagen numpy array
            points: puntos de prompt [[x1, y1], [x2, y2], ...]
            boxes: bounding boxes [[x1, y1, x2, y2], ...]
            labels: labels para puntos (1=foreground, 0=background)
        """
        # Convertir frame a formato adecuado
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Convertir BGR a RGB si es necesario
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Ejecutar predicción
        results = self.model.predict(
            source=frame,
            points=points,
            bboxes=boxes,
            labels=labels,
            verbose=False
        )
        
        return results
    
    def process_with_prompts(self, frame, prompts):
        """
        Procesar con prompts específicos
        prompts: lista de dicts con 'points', 'boxes', o 'labels'
        """
        return self.process_frame(frame, **prompts)

# Alternativa para segmentación automática (sin prompts)
class UltralyticsSAMAuto:
    def __init__(self, model_path="sam2.1_b.pt"):
        from ultralytics import SAM
        self.model = SAM(model_path)
    
    def process_auto_segment(self, frame):
        """Segmentación automática de todo en la imagen"""
        results = self.model.predict(source=frame, verbose=False)
        return results