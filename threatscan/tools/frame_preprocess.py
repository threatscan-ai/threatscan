import cv2

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_bgr2rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    return frame_bgr2rgb

def preprocess_frames(frames):
    preprocessed_frames = []
    for frame in frames:
        preprocessed_frame = preprocess_frame(frame)
        preprocessed_frames.append(preprocessed_frame)
    return preprocessed_frames
