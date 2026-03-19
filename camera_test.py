import cv2
import time

from src.camera_capture import select_best_capture

# Force reasonable resolution
capture_size = (320, 240)
cap, backend_name, measured_fps = select_best_capture(
    camera_index=0,
    capture_size=capture_size,
    target_fps=30,
    benchmark=False,
)

if cap is None or not cap.isOpened():
    print("Camera could not be opened")
    raise SystemExit(1)

print(f"Using backend {backend_name}")

prev_time = time.time()
fps_sum = 0.0
fps_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    fps_sum += fps
    fps_count += 1
    avg_fps = fps_sum / fps_count

    cv2.putText(frame, f"{backend_name} FPS: {int(fps)} Avg: {int(avg_fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Camera Test", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()