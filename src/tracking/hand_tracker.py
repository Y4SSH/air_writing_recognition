import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional


class HandTracker:
    """
    Real-time hand tracking and index fingertip extraction using MediaPipe.
    Optimized for stability and air-writing UX.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Initialize MediaPipe Hands with stable real-time configuration.
        """

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
        """
        Process frame to detect hand landmarks and extract index fingertip.

        Returns:
            processed_frame (np.ndarray)
            normalized_coordinates (Optional[Tuple[float, float]])
        """

        # Defensive frame validation
        if frame is None or frame.size == 0:
            return frame, None

        # Mirror frame for natural air-writing UX
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Performance optimization: pass by reference
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)

        normalized_coordinates = None

        if results.multi_hand_landmarks:
            # Since max_num_hands = 1, directly access first element
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract index fingertip safely using enum
            index_tip = hand_landmarks.landmark[
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]

            normalized_coordinates = (index_tip.x, index_tip.y)

            # Optional: draw fingertip highlight
            h, w, _ = frame.shape
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

        return frame, normalized_coordinates

    def release(self) -> None:
        """
        Release MediaPipe resources explicitly.
        """
        self.hands.close()