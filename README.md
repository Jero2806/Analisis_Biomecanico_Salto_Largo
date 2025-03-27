# 🦵 Análisis Biomecánico del Salto Largo

Este repositorio contiene el desarrollo completo de un sistema de análisis biomecánico aplicado al gesto técnico del **salto largo sin impulso**, utilizando visión por computadora y análisis cuantitativo de datos espaciales.

El sistema fue desarrollado en Python y utiliza [MediaPipe](https://google.github.io/mediapipe/) para la detección automática de puntos clave del cuerpo (pose estimation).

---

## 🎯 Objetivo

Evaluar la calidad del gesto de salto largo a través de:

- Extracción automática de coordenadas articulares (cadera, rodilla, tobillo).
- Cálculo de **trayectorias**, **velocidades**, **aceleraciones** y **ángulo de la rodilla**.
- Comparación entre diferentes ejecuciones para identificar errores técnicos.
- Exportación de datos y visualización con gráficas organizadas.

---

## 📂 Estructura del Repositorio

Analisis_Biomecanico/
├── Code/
│   ├── Recolección_Datos.py         # Código principal que analiza un solo salto
│   └── Comparacion_Datos.py         # Código que compara métricas entre los 4 videos
├── data/                            # Datos exportados como archivos .csv
│   ├── Trayectorias{1-4}.csv
│   ├── Velocidades{1-4}.csv
│   ├── Aceleraciones{1-4}.csv
│   └── angulo_rodilla{1-4}.csv
├── Graphs/
│   ├── Comparisson Graphs/          # Gráficas comparativas entre los 4 videos
│   └── Individual Graphs/           # Gráficas individuales por cada ejecución
├── salto_largo{1-4}.mp4             # Videos analizados
└── README.md                        # Este archivo

---

## 🎥 Descripción de los Videos

- salto_largo1.mp4, 2.mp4, 3.mp4: Ejecuciones con errores técnicos.
- salto_largo4.mp4: Ejecución de referencia con técnica adecuada (control).

---

## 🧠 ¿Qué hace cada archivo de código?

### Recolección_Datos.py
- Carga un video de salto.
- Detecta puntos clave del cuerpo (MediaPipe Pose).
- Extrae coordenadas de cadera, rodilla y tobillo.
- Calcula:
  - Trayectorias en 2D.
  - Velocidades y aceleraciones (con filtro pasa-bajas).
  - Ángulo de la rodilla en cada frame.
- Exporta los resultados como archivos .csv.
- Genera gráficas de análisis individuales.

### Comparacion_Datos.py
- Carga los .csv de los 4 saltos.
- Asocia cada frame con una distancia (basado en la distancia total medida en cada salto).
- Compara gráficamente entre los 4 saltos:
  - Trayectoria articular
  - Velocidades (cadera, rodilla, tobillo)
  - Aceleraciones (cadera, rodilla, tobillo)

---

## 📊 Gráficas Generadas

### Graphs/Comparisson Graphs/
- Gráficas superpuestas de los 4 videos:
  - Trayectorias_Distancia.png
  - Velocidad_*_Distancia.png
  - Aceleracion_*_Distancia.png
  - También incluye gráficos por frames

### Graphs/Individual Graphs/
- Gráficas por video (1-4) y por métrica:
  - Trayectoria{i}.png: Trayectoria articular (Y vs X)
  - Velocidad{i}.png: Velocidades articulares
  - Aceleracion{i}.png: Aceleraciones articulares
  - Angulo{i}.png: Ángulo de la rodilla a lo largo del tiempo

---

## 📈 Archivos .csv en data/

Cada archivo contiene los datos procesados para análisis adicional o visualización externa:

- Trayectorias*.csv: Posiciones X, Y de cadera, rodilla, tobillo.
- Velocidades*.csv: Velocidades lineales por articulación.
- Aceleraciones*.csv: Aceleraciones por articulación.
- angulo_rodilla*.csv: Evolución del ángulo de la rodilla.

---

## 🛠 Requisitos

- Python 3.7+
- OpenCV (cv2)
- MediaPipe
- NumPy, Pandas, Matplotlib
- SciPy (para filtro pasa-bajas)

---

## 📌 Notas Finales

- Asegúrate de tener una referencia de tamaño en los videos para interpretar la distancia.
- El video debe estar grabado en formato horizontal, sin cortes, y debe capturar todas las fases del salto.
- El Video 4 se considera la ejecución modelo o técnica correcta.

---

¿Preguntas? ¿Ideas para mejorar? ¡Bienvenido a contribuir!
"""
