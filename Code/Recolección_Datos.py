# ========================================
#         ANÁLISIS BIOMECÁNICO DEL SALTO
# ========================================

# === Librerías necesarias ===
import cv2
import mediapipe as mp
import numpy as np
import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.signal import butter, filtfilt

# === Cargar video ===
video_path = "salto_largo2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el video, revisa la ruta")
else:
    print("Video cargado correctamente")

# === Inicializar MediaPipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# === Mostrar video con pose estimada en tiempo real (solo Jupyter) ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    display.display(img)
    display.clear_output(wait=True)

cap.release()

# === Función para obtener coordenadas de un punto clave ===
def get_landmark_position(landmarks, index):
    if landmarks:
        return (landmarks.landmark[index].x, landmarks.landmark[index].y)
    return None

# === Segunda pasada: almacenar coordenadas brutas ===
cap = cv2.VideoCapture(video_path)
left_hip, right_hip = [], []
left_knee, right_knee = [], []
left_ankle, right_ankle = [], []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        lh = get_landmark_position(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        rh = get_landmark_position(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
        lk = get_landmark_position(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        rk = get_landmark_position(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
        la = get_landmark_position(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        ra = get_landmark_position(results.pose_landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)

        if lh and rh and lk and rk and la and ra:
            left_hip.append(lh)
            right_hip.append(rh)
            left_knee.append(lk)
            right_knee.append(rk)
            left_ankle.append(la)
            right_ankle.append(ra)

cap.release()

# === Filtro pasa bajas ===
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

cutoff = 3
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0.0:
    fps = 30
    print("⚠️ Se asumió 30 FPS por defecto")

# === Aplicar filtro a los datos ===
def filter_coords(coords):
    return np.column_stack((
        butter_lowpass_filter([p[0] for p in coords], cutoff, fps),
        butter_lowpass_filter([p[1] for p in coords], cutoff, fps)
    ))

left_hip = filter_coords(left_hip)
right_hip = filter_coords(right_hip)
left_knee = filter_coords(left_knee)
right_knee = filter_coords(right_knee)
left_ankle = filter_coords(left_ankle)
right_ankle = filter_coords(right_ankle)

# === Recolectar puntos para trayectorias de cadera, rodilla y tobillo izquierdos ===
hip_x, hip_y = [], []
knee_x, knee_y = [], []
ankle_x, ankle_y = [], []

cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        left_hip = get_landmark_position(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        left_knee = get_landmark_position(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        left_ankle = get_landmark_position(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)

        if left_hip: hip_x.append(left_hip[0]); hip_y.append(left_hip[1])
        if left_knee: knee_x.append(left_knee[0]); knee_y.append(left_knee[1])
        if left_ankle: ankle_x.append(left_ankle[0]); ankle_y.append(left_ankle[1])

cap.release()

# === Gráfico de trayectorias ===
plt.figure(figsize=(8, 6))
plt.plot(hip_x, hip_y, label="Cadera", color='blue')
plt.plot(knee_x, knee_y, label="Rodilla", color='green')
plt.plot(ankle_x, ankle_y, label="Tobillo", color='red')
plt.xlabel("Posición X (normalizada)")
plt.ylabel("Posición Y (normalizada)")
plt.title("Trayectoria de los Puntos Clave Durante el Salto")
plt.legend()
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# === Calcular velocidad ===
def calculate_velocity(x_coords, y_coords, time_step):
    return [0] + [np.linalg.norm([x_coords[i] - x_coords[i-1], y_coords[i] - y_coords[i-1]]) / time_step
                  for i in range(1, len(x_coords))]

time_step = 1 / 30
hip_velocity = calculate_velocity(hip_x, hip_y, time_step)
knee_velocity = calculate_velocity(knee_x, knee_y, time_step)
ankle_velocity = calculate_velocity(ankle_x, ankle_y, time_step)

# === Graficar velocidad ===
frame_numbers = list(range(len(hip_velocity)))
hip_velocity = butter_lowpass_filter(hip_velocity, cutoff, fps)
knee_velocity = butter_lowpass_filter(knee_velocity, cutoff, fps)
ankle_velocity = butter_lowpass_filter(ankle_velocity, cutoff, fps)

plt.figure(figsize=(10, 6))
plt.plot(frame_numbers, hip_velocity, label="Velocidad Cadera", color='blue')
plt.plot(frame_numbers, knee_velocity, label="Velocidad Rodilla", color='green')
plt.plot(frame_numbers, ankle_velocity, label="Velocidad Tobillo", color='red')
plt.xlabel("Frame")
plt.ylabel("Velocidad (unidades/s)")
plt.title("Velocidad de los Puntos Clave Durante el Salto")
plt.legend()
plt.grid(True)
plt.show()

# === Calcular aceleración ===
def calculate_acceleration(velocity, time_step):
    return [0] + [(velocity[i] - velocity[i-1]) / time_step for i in range(1, len(velocity))]

hip_acceleration = calculate_acceleration(hip_velocity, time_step)
knee_acceleration = calculate_acceleration(knee_velocity, time_step)
ankle_acceleration = calculate_acceleration(ankle_velocity, time_step)

hip_acceleration = butter_lowpass_filter(hip_acceleration, cutoff, fps)
knee_acceleration = butter_lowpass_filter(knee_acceleration, cutoff, fps)
ankle_acceleration = butter_lowpass_filter(ankle_acceleration, cutoff, fps)

# === Graficar aceleración ===
plt.figure(figsize=(10, 6))
plt.plot(frame_numbers, hip_acceleration, label="Aceleración Cadera", color='blue')
plt.plot(frame_numbers, knee_acceleration, label="Aceleración Rodilla", color='green')
plt.plot(frame_numbers, ankle_acceleration, label="Aceleración Tobillo", color='red')
plt.xlabel("Frame")
plt.ylabel("Aceleración (unidades/s²)")
plt.title("Aceleración de los Puntos Clave Durante el Salto")
plt.legend()
plt.grid(True)
plt.show()

# === Calcular ángulo de la rodilla ===
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

frame_numbers = []
knee_angles = []
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        hip = get_landmark_position(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        knee = get_landmark_position(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        ankle = get_landmark_position(results.pose_landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)

        if hip and knee and ankle:
            angle = calculate_angle(hip, knee, ankle)
            frame_numbers.append(frame_count)
            knee_angles.append(angle)

    frame_count += 1

cap.release()

# === Graficar ángulo de rodilla ===
plt.figure(figsize=(10, 5))
plt.plot(frame_numbers, knee_angles)
plt.xlabel("Número de Frame")
plt.ylabel("Ángulo de la Rodilla (°)")
plt.title("Evolución del Ángulo de la Rodilla Durante el Salto")
plt.grid(True)
plt.show()

# === Exportar a CSV ===
output_folder = "C:\\Users\\user\\Downloads\\Universidad\\Octavo Semestre\\Biomecanica\\Analisis_Biomecanico\\data"
os.makedirs(output_folder, exist_ok=True)

pd.DataFrame({"Frame": frame_numbers, "Ángulo Rodilla (°)": knee_angles}).to_csv(f"{output_folder}/angulo_rodilla4.csv", index=False)
pd.DataFrame({
    "Frame": frame_numbers,
    "Cadera_X": hip_x,
    "Cadera_Y": hip_y,
    "Rodilla_X": knee_x,
    "Rodilla_Y": knee_y,
    "Tobillo_X": ankle_x,
    "Tobillo_Y": ankle_y
}).to_csv(f"{output_folder}/Trayectorias4.csv", index=False)
pd.DataFrame({
    "Frame": frame_numbers,
    "Velocidad Cadera": hip_velocity,
    "Velocidad Rodilla": knee_velocity,
    "Velocidad Tobillo": ankle_velocity
}).to_csv(f"{output_folder}/Velocidades4.csv", index=False)
pd.DataFrame({
    "Frame": frame_numbers,
    "Aceleración Cadera": hip_acceleration,
    "Aceleración Rodilla": knee_acceleration,
    "Aceleración Tobillo": ankle_acceleration
}).to_csv(f"{output_folder}/Aceleraciones4.csv", index=False)

print("✅ Archivos CSV exportados correctamente.")
