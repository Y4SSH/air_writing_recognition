import cv2
import time
import threading
from queue import Empty, Full, Queue
import logging

from src.camera_capture import select_best_capture
from src.tracking.hand_tracker import HandTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():

    # Ultra-fast preset: prioritize smooth UI FPS over landmark update rate.
    capture_size = (320, 240)
    tracker = HandTracker(processing_size=(128, 96))
    target_inference_fps = 8.0
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
        if coords is not None:

            if smooth_coords is None:
                smooth_coords = coords
            else:
                sx, sy = smooth_coords
                cx, cy = coords

                smooth_coords = (
                    alpha * cx + (1 - alpha) * sx,
                    alpha * cy + (1 - alpha) * sy
                )

        # draw fingertip
        if smooth_coords is not None:
            h, w, _ = frame.shape
            px = int(smooth_coords[0] * w)
            py = int(smooth_coords[1] * h)

            cv2.circle(frame, (px, py), 10, (255, 0, 0), -1)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Air Writing Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # cleanup
    running["flag"] = False
    thread.join(timeout=1)

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()