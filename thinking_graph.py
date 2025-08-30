import cv2
from ultralytics import YOLO
import networkx as nx
import requests  # para LLM API

# 1. Cargar YOLO
model = YOLO("yolov8n.pt")

# 2. Crear grafo
G = nx.DiGraph()
G.add_node("inicio")
G.add_node("persona_detectada")
G.add_edge("inicio", "persona_detectada", condition="persona en cam")

# 3. LLM API
def llm_decision(context):
    prompt = f"Contexto: {context}. ¿Qué debería hacer el agente G1?"
    response = requests.post("https://api.openai.com/v1/chat/completions", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}]
    }, headers={"Authorization": "Bearer ..."})
    return response.json()["choices"][0]["message"]["content"]

# 4. Loop principal
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = model(frame)
    detections = results[0].boxes  # objetos detectados

    # Extraer clases
    detected_classes = [model.names[int(box.cls)] for box in detections]

    # Actualizar grafo según detecciones
    if "person" in detected_classes:
        G1_state = "persona_detectada"

        # Consultar LLM
        context = f"Objetos: {detected_classes}"
        suggestion = llm_decision(context)
        print("LLM sugiere:", suggestion)

        # Tomar acción (ej: hablar, moverse)
        if "ofrecer" in suggestion.lower():
            g1.ejecutar_accion("hablar", "¿Puedo ayudarle?")