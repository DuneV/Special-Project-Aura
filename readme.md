# 🤖 Zero-Shot Learning for Unitree G1 Locomotion & Manipulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org)
[![MuJoCo 3.0](https://img.shields.io/badge/MuJoCo-3.0-orange)](https://mujoco.org)

> **Implementación práctica de Zero-Shot Learning para el robot humanoide Unitree G1, permitiendo ejecutar tareas de locomoción y manipulación mediante instrucciones en lenguaje natural sin entrenamiento específico.**

## 🌟 Visión
Este proyecto implementa un sistema de **aprendizaje cero disparo (ZSL)** para el robot humanoide **Unitree G1**, permitiendo que el robot ejecute tareas mediante instrucciones en lenguaje natural sin entrenamiento específico para cada tarea. La solución integra representaciones semánticas del lenguaje con control robótico en simulación (MuJoCo) y hardware real.

## ⚙️ ¿Qué es CLIP y por qué es clave para este proyecto?

**CLIP (Contrastive Language-Image Pretraining)** es un modelo de inteligencia artificial desarrollado por OpenAI que aprende asociaciones entre imágenes y texto durante su entrenamiento. Es fundamental para nuestro proyecto por estas razones:

### 🔍 ¿Cómo funciona CLIP?
- **Entrenamiento contrastivo**: Aprende a asociar imágenes con sus descripciones de texto en un espacio vectorial compartido
- **Dos componentes principales**:
  1. **Image Encoder**: Convierte imágenes en vectores (embeddings)
  2. **Text Encoder**: Convierte texto en vectores del mismo espacio
- **Cero disparo (zero-shot)**: Puede clasificar imágenes en categorías no vistas durante el entrenamiento

### 💡 Ejemplo práctico en nuestro proyecto:
```python
import clip
import torch

# Cargar modelo preentrenado
model, preprocess = clip.load("ViT-B/32")

# Convertir comando de voz a vector
command = "camina hacia el objeto rojo"
text_input = clip.tokenize([command]).to(device)
text_features = model.encode_text(text_input)

# Convertir imagen de la cámara del robot a vector
image = preprocess(pil_image).unsqueeze(0).to(device)
image_features = model.encode_image(image)

# Calcular similitud entre comando e imagen
similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)