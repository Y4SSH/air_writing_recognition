from typing import List, Optional, Tuple


class TrajectoryTracker:
    def __init__(self, max_missing: int = 8, min_points: int = 15):
        self.points: List[Tuple[float, float]] = []
        self.missing_frames: int = 0
        self.max_missing: int = max_missing
        self.min_points: int = min_points

    def update(self, coords: Optional[Tuple[float, float]]) -> None:
        if coords is None:
            self.missing_frames += 1
            return

        self.missing_frames = 0
        self.points.append(coords)

    def stroke_finished(self) -> bool:
        return self.missing_frames > self.max_missing

    def is_valid_stroke(self) -> bool:
        return len(self.points) > self.min_points

    def get_points(self) -> List[Tuple[float, float]]:
        return self.points

    def reset(self) -> None:
        self.points = []
        self.missing_frames = 0
