import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Tuple, Optional

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    """
    Lightweight fingertip tracker using MediaPipe Tasks HandLandmarker.
    Only returns the index fingertip coordinate.
    """

    def __init__(
        self,
        model_asset_path: str = "models/hand_landmarker.task",
        processing_size: Optional[Tuple[int, int]] = (192, 144),
        max_num_hands: int = 1,
        min_hand_detection_confidence: float = 0.75,
        min_hand_presence_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7
    ):

        self.processing_size = processing_size

        base_options = python.BaseOptions(model_asset_path=model_asset_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        # required for VIDEO mode
        self.last_timestamp_ms = 0

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:

        if frame is None or frame.size == 0:
            return frame, None

        # Run detection on a smaller frame for speed when configured.
        if self.processing_size is not None:
            if frame.shape[1] != self.processing_size[0] or frame.shape[0] != self.processing_size[1]:
                small_frame = cv2.resize(frame, self.processing_size)
            else:
                small_frame = frame
        else:
            small_frame = frame

        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # ensure increasing timestamps
        timestamp_ms = int(time.monotonic() * 1000)

        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1

        self.last_timestamp_ms = timestamp_ms

        results = self.detector.detect_for_video(mp_image, timestamp_ms)

        coords = None

        if results.hand_landmarks:

            hand_landmarks = results.hand_landmarks[0]

            # index fingertip = landmark 8
            index_tip = hand_landmarks[8]

            coords = (index_tip.x, index_tip.y)

        return frame, coords

    def release(self):
        self.detector.close()