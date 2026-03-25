import cv2
import time
import threading
from queue import Empty, Full, Queue
import logging

from src.camera_capture import select_best_capture
from src.trajectory.trajectory_tracker import TrajectoryTracker
from src.tracking.hand_tracker import HandTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():

    # Balanced preset: improved accuracy with smooth rendering.
    capture_size = (320, 240)
    tracker = HandTracker(processing_size=(192, 144))
    target_inference_fps = 12.0
    inference_interval = 1.0 / target_inference_fps

    cap, backend_name, measured_fps = select_best_capture(
        camera_index=0,
        capture_size=capture_size,
        target_fps=30,
        benchmark=False,
    )

    if cap is None or not cap.isOpened():
        logging.error("Camera could not be opened")
        return

    logging.info("Using camera backend %s", backend_name)

    # shared data
    frame_queue = Queue(maxsize=1)
    latest_coords = {"value": None}
    running = {"flag": True}

    # smoothing
    smooth_coords = None
    alpha = 0.85

    trajectory = TrajectoryTracker()
    dataset = []
    tracking_paused = False

    def inference_loop():
        while running["flag"]:
            try:
                frame = frame_queue.get(timeout=0.02)
            except Empty:
                continue

            _, coords = tracker.process_frame(frame)
            latest_coords["value"] = coords

    # start inference thread
    thread = threading.Thread(target=inference_loop, daemon=True)
    thread.start()

    prev_time = time.time()
    last_infer_enqueue_time = time.perf_counter()

    logging.info("Press 'q' to quit")

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Throttle inference and avoid queue churn to keep render FPS higher.
        now = time.perf_counter()
        if now - last_infer_enqueue_time >= inference_interval:
            infer_frame = cv2.resize(frame, tracker.processing_size)
            try:
                frame_queue.put_nowait(infer_frame)
            except Full:
                pass
            last_infer_enqueue_time = now

        coords = latest_coords["value"]

        # smoothing
        if coords is not None and not tracking_paused:

            if smooth_coords is None:
                smooth_coords = coords
            else:
                sx, sy = smooth_coords
                cx, cy = coords

                smooth_coords = (
                    alpha * cx + (1 - alpha) * sx,
                    alpha * cy + (1 - alpha) * sy
                )
        elif tracking_paused:
            smooth_coords = None
        else:
            smooth_coords = None

        if tracking_paused:
            trajectory.update(None)
        else:
            trajectory.update(smooth_coords)

        if trajectory.stroke_finished():
            if trajectory.is_valid_stroke():
                stroke = trajectory.get_points()
                dataset.append(stroke)
                print(f"Stroke saved | Total: {len(dataset)} | Points: {len(stroke)}")
            trajectory.reset()

        # draw fingertip
        if smooth_coords is not None:
            h, w, _ = frame.shape
            px = int(smooth_coords[0] * w)
            py = int(smooth_coords[1] * h)

            cv2.circle(frame, (px, py), 10, (255, 0, 0), -1)

        path_points = trajectory.get_points()
        if len(path_points) > 1:
            h, w, _ = frame.shape
            for i in range(1, len(path_points)):
                x1 = int(path_points[i - 1][0] * w)
                y1 = int(path_points[i - 1][1] * h)
                x2 = int(path_points[i][0] * w)
                y2 = int(path_points[i][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Strokes: {len(dataset)}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        status_text = "PAUSED" if tracking_paused else "TRACKING"
        status_color = (0, 0, 255) if tracking_paused else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow("Air Writing Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            trajectory.reset()
        if key == ord("s"):
            if trajectory.is_valid_stroke():
                stroke = trajectory.get_points()
                dataset.append(stroke)
                print(f"Stroke saved | Total: {len(dataset)} | Points: {len(stroke)}")
            trajectory.reset()
            tracking_paused = False
        if key == 32:  # spacebar
            tracking_paused = not tracking_paused

    # cleanup
    running["flag"] = False
    thread.join(timeout=1)

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()