import cv2
import numpy as np

# Parametry wideo
height, width = 720, 1280
num_frames = 100
fps = 30
output_filename = "output_video.mp4"

# Generowanie klatek wideo (gradient)
frames = [(np.linspace(0, 1, width).astype(np.float64) * (i / num_frames)).reshape(1, width).repeat(height, axis=0)
          for i in range(num_frames)]

# Konwersja klatek na uint8 (skalowanie do 0-255)
frames = [(frame * 255).astype(np.uint8) for frame in frames]

# OpenCV VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Kodek MP4
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Zapis klatek do wideo
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))  # Konwersja do BGR (OpenCV wymaga kolorowego wejścia)

out.release()
print(f"Plik zapisano jako {output_filename}")
