from System import System
import cv2

system = System('TestVideos/MOT16-01-raw.webm')

new_capture = system.run_with_hungarian()

height, width = new_capture[0].shape[:2]
output_filename = "output_video.mp4"
fps = 10
# OpenCV VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Saving frames to file
for frame in new_capture:
    out.write(frame)

out.release()
print(f"Plik zapisano jako {output_filename}")
