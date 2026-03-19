import time
from typing import List, Optional, Tuple

import cv2


def _backend_name(backend: int) -> str:
    if backend == cv2.CAP_DSHOW:
        return "CAP_DSHOW"
    if backend == cv2.CAP_MSMF:
        return "CAP_MSMF"
    if backend == cv2.CAP_ANY:
        return "CAP_ANY"
    return f"CAP_{backend}"


def _open_capture(
    camera_index: int,
    backend: int,
    capture_size: Tuple[int, int],
) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def _benchmark_capture(cap: cv2.VideoCapture, warmup_frames: int = 8, sample_frames: int = 45) -> float:
    # Warm up camera buffers before timing.
    for _ in range(warmup_frames):
        ok, _ = cap.read()
        if not ok:
            return 0.0

    start = time.perf_counter()
    frames = 0
    for _ in range(sample_frames):
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        return 0.0
    return frames / elapsed


def select_best_capture(
    camera_index: int = 0,
    capture_size: Tuple[int, int] = (320, 240),
    target_fps: int = 30,
    candidate_backends: Optional[List[int]] = None,
    benchmark: bool = False,
) -> Tuple[Optional[cv2.VideoCapture], str, float]:
    if candidate_backends is None:
        candidate_backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]

    best_cap: Optional[cv2.VideoCapture] = None
    best_name = "NONE"
    best_fps = 0.0

    for backend in candidate_backends:
        cap = _open_capture(camera_index, backend, capture_size)
        if cap is None:
            continue

        # Use first opened backend in non-benchmark mode to avoid startup stalls.
        if not benchmark:
            return cap, _backend_name(backend), 0.0

        measured_fps = _benchmark_capture(cap)

        if measured_fps > best_fps:
            if best_cap is not None:
                best_cap.release()
            best_cap = cap
            best_fps = measured_fps
            best_name = _backend_name(backend)
        else:
            cap.release()

    return best_cap, best_name, best_fps
