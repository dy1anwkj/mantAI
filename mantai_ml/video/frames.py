import cv2


def iter_frames(path, sample_fps=2.0):
    """Yield (seconds, frame) pairs, about sample_fps per second of video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps))) if sample_fps else 1
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                yield index / fps, frame
            index += 1
    finally:
        cap.release()


def crop_roi(frame, roi):
    """roi = [x, y, w, h] as fractions of the frame (0-1). None = whole frame.
    Returns (crop, (x0, y0)) so boxes can be mapped back to the frame."""
    if roi is None:
        return frame, (0, 0)
    H, W = frame.shape[:2]
    x, y, w, h = roi
    x0, y0 = int(x * W), int(y * H)
    x1, y1 = int((x + w) * W), int((y + h) * H)
    return frame[y0:y1, x0:x1], (x0, y0)
