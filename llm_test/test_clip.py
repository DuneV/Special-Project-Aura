""" from transformers import pipeline

# Clasificación zero-shot
classifier = pipeline("zero-shot-classification")
result = classifier(
    "Me gusta mucho esta película",
    candidate_labels=["positivo", "negativo", "neutral"]
)
print(result)
 """

import clip
import torch
from PIL import Image
import os
import numpy as np

# Constants

dir_path = os.path.dirname(os.path.realpath(__file__))
model, preprocess = clip.load("ViT-B/32")


# Cargar imagen
image = preprocess(Image.open(f"{dir_path}/glass.jpeg")).unsqueeze(0)

# Definir clases nunca vistas en entrenamiento
text_queries = ["un gato", "un perro", "un coche", "una casa", "un vaso", "un tea"]
text_tokens = clip.tokenize(text_queries)

# Hacer predicción
with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text_tokens)
    probs = logits_per_image.softmax(dim=-1)

print(np.array(probs).argmax)
