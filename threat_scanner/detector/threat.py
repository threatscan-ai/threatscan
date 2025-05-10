import cv2
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from threat_scanner.tools.frame_preprocess import preprocess_frame

class ThreatDetector:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = VideoMAEImageProcessor.from_pretrained(self.model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.num_frames_to_predict = 17
        self.frame_diff_threshold = 0.1
    
    def predict(self, frames):
        if not frames:
            return None
        inputs = self.processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probs, dim=-1).item()
        predicted_class_name = self.model.config.id2label[predicted_class_idx]
        predicted_class_prob = probs[0][predicted_class_idx].item()
        return (predicted_class_name, predicted_class_prob)
    
    def scan(self, source):
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error opening video file")
            return
        
        
        frames = []
        previous_frame = None
        prediction = None
        probability = None
        index = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                processed_frame = preprocess_frame(frame.copy())
                if previous_frame is not None:
                    diff = cv2.absdiff(previous_frame, processed_frame)
                    mean_diff = diff.mean()
                    if mean_diff > self.frame_diff_threshold:
                        frames.append(processed_frame)
                        if len(frames) == self.num_frames_to_predict:
                            prediction, probability = self.predict(frames)
                            frames = []
                else:
                    frames.append(processed_frame)
                    if len(frames) == self.num_frames_to_predict:
                        prediction, probability = self.predict(frames)
                        frames = []
                if prediction and probability:
                    cv2.putText(frame, f"{prediction} {probability}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                cv2.imshow('Frame', frame)

                # Wait for 25 milliseconds for a key press. If 'q' is pressed, exit the loop
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            else:
                break
            previous_frame = processed_frame
            index += 1
        print("Done")
        cap.release()
        cv2.destroyAllWindows()
