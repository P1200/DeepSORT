import json
import os

from System import System
import cv2
import sys

if len(sys.argv) != 3:
    print("Two arguments are required - input file and output filename.")
    sys.exit(1)

video_path = sys.argv[1]
output_filename = sys.argv[2]
system = System(video_path)

new_capture, history = system.run_with_hungarian()

history_filename = output_filename + ".json"

# if os.path.exists(history_filename):
#     with open(history_filename, "w") as f:
#         json.dump([], f)
#
# # Append to the JSON file
# try:
#     with open(history_filename, "r") as f:
#         data = json.load(f)
# except (FileNotFoundError, json.JSONDecodeError):
#     data = []
#
# data.extend(history)

with open(history_filename, "w") as f:
    json.dump(history, f, indent=4)
print("History saved in: " + history_filename)

height, width = new_capture[0].shape[:2]
output_filename = output_filename + ".mp4"
fps = 10
# OpenCV VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Saving frames to file
for frame in new_capture:
    out.write(frame)

out.release()
print(f"File saved as {output_filename}")
