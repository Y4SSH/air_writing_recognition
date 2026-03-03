import sys
import cv2
import time
import logging
from src.tracking.hand_tracker import HandTracker

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main():
    tracker = None
    cap = None
    last_coords = None  # Dropout buffer

    try:
        # Initialize HandTracker
        tracker = HandTracker()

        # Open webcam (Windows optimized)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            logging.warning("CAP_DSHOW failed. Trying default backend...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logging.error("Failed to open webcam.")
                return

        logging.info("Webcam opened successfully. Press 'q' to exit.")

        prev_time = time.time()

        while True:
            success, frame = cap.read()
            if not success:
                logging.warning("Failed to grab frame.")
                break

            # IMPORTANT: Do NOT flip here
            # Flipping is already handled inside HandTracker

            processed_frame, coords = tracker.process_frame(frame)

            # Dropout stabilization
            if coords is not None:
                last_coords = coords
            else:
                coords = last_coords

            # FPS calculation
            curr_time = time.time()
            delta = curr_time - prev_time
            fps = 1 / delta if delta > 0 else 0
            prev_time = curr_time

            cv2.putText(
                processed_frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Draw fingertip indicator (if available)
            if coords is not None:
                x, y = coords
                h, w, _ = processed_frame.shape
                px, py = int(x * w), int(y * h)
                cv2.circle(processed_frame, (px, py), 10, (255, 0, 0), cv2.FILLED)

            cv2.imshow("Air-Writing Recognition - Hand Tracking", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exit initiated by user.")
                break

    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    finally:
        logging.info("Releasing resources...")
        if cap is not None:
            cap.release()
        if tracker is not None:
            tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()