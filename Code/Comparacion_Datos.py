# =============================================
#     COMPARACIÓN DE MÉTRICAS ENTRE VIDEOS
# =============================================

import pandas as pd
import matplotlib.pyplot as plt
import os

# === Nombres de las versiones/videos ===
versiones = ["1", "2", "3", "4"]
base_path = "data"  # Carpeta donde están almacenados los CSV

# === Cargar los archivos en diccionarios separados ===
trayectorias = {}
angulos = {}
velocidades = {}
aceleraciones = {}

for v in versiones:
    trayectorias[v] = pd.read_csv(os.path.join(base_path, f"Trayectorias{v}.csv"))
    angulos[v] = pd.read_csv(os.path.join(base_path, f"angulo_rodilla{v}.csv"))
    velocidades[v] = pd.read_csv(os.path.join(base_path, f"Velocidades{v}.csv"))
    aceleraciones[v] = pd.read_csv(os.path.join(base_path, f"Aceleraciones{v}.csv"))

# === Distancia total del salto en cada video (estimada o medida en metros) ===
distancias_totales = {
    "1": 2.00,
    "2": 1.63,
    "3": 1.78,
    "4": 2.40
}

# === Función para añadir columna de distancia en cada DataFrame ===
def añadir_columna_distancia(diccionario, nombre_diccionario):
    for v in versiones: 
        total_frames = len(diccionario[v])
        paso = distancias_totales[v] / (total_frames - 1)  # Asume distancia uniforme por frame
        distancia = [i * paso for i in range(total_frames)]
        diccionario[v]["Distancia"] = distancia
        print(f"✅ Distancia añadida a {nombre_diccionario} del video {v}")

# === Añadir la columna "Distancia" a cada grupo de datos ===
añadir_columna_distancia(aceleraciones, "aceleraciones")
añadir_columna_distancia(velocidades, "velocidades")
añadir_columna_distancia(angulos, "angulos")
añadir_columna_distancia(trayectorias, "trayectorias")

# === Graficar trayectorias articulares en función de la distancia ===
fig, axs = plt.subplots(1, 4, figsize=(18, 5))
for i, v in enumerate(versiones):
    df = trayectorias[v]
    axs[i].plot(df['Distancia'], df['Cadera_Y'], label='Cadera')
    axs[i].plot(df['Distancia'], df['Rodilla_Y'], label='Rodilla')
    axs[i].plot(df['Distancia'], df['Tobillo_Y'], label='Tobillo')
    axs[i].invert_yaxis()
    axs[i].set_title(f"Trayectoria Video {v}")
    axs[i].set_xlabel("Distancia (m)")
    axs[i].set_ylabel("Posición Y")
    axs[i].legend()
plt.tight_layout()
plt.show()

# === Velocidades por segmento en función de la distancia ===
plt.figure(figsize=(12, 6))
for v in versiones:
    plt.plot(velocidades[v]['Distancia'], velocidades[v]['Velocidad Rodilla'], label=f'Video {v}')
plt.title("Velocidad de la Rodilla en 2D")
plt.xlabel("Distancia (m)")
plt.ylabel("Velocidad")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for v in versiones:
    plt.plot(velocidades[v]['Distancia'], velocidades[v]['Velocidad Tobillo'], label=f'Video {v}')
plt.title("Velocidad del Tobillo en 2D")
plt.xlabel("Distancia (m)")
plt.ylabel("Velocidad")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for v in versiones:
    plt.plot(velocidades[v]['Distancia'], velocidades[v]['Velocidad Cadera'], label=f'Video {v}')
plt.title("Velocidad de la Cadera en 2D")
plt.xlabel("Distancia (m)")
plt.ylabel("Velocidad")
plt.legend()
plt.grid(True)
plt.show()

# === Aceleraciones por segmento en función de la distancia ===
plt.figure(figsize=(12, 6))
for v in versiones:
    plt.plot(aceleraciones[v]['Distancia'], aceleraciones[v]['Aceleración Rodilla'], label=f'Video {v}')
plt.title("Aceleración de la Rodilla en 2D")
plt.xlabel("Distancia (m)")
plt.ylabel("Aceleración")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for v in versiones:
    plt.plot(aceleraciones[v]['Distancia'], aceleraciones[v]['Aceleración Tobillo'], label=f'Video {v}')
plt.title("Aceleración del Tobillo en 2D")
plt.xlabel("Distancia (m)")
plt.ylabel("Aceleración")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for v in versiones:
    plt.plot(aceleraciones[v]['Distancia'], aceleraciones[v]['Aceleración Cadera'], label=f'Video {v}')
plt.title("Aceleración de la Cadera en 2D")
plt.xlabel("Distancia (m)")
plt.ylabel("Aceleración")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 6))

# Paleta de colores para cada video
colores = ['blue', 'orange', 'green','red']

# Graficar todos los ángulos juntos
for i, v in enumerate(versiones):
    plt.plot(angulos[v]["Frame"], angulos[v]["Ángulo Rodilla (°)"], 
             label=f"Video {v}", color=colores[i])

plt.title("Comparación del Ángulo de la Rodilla entre los Cuatro Videos")
plt.xlabel("Frame")
plt.ylabel("Ángulo (°)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()