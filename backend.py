import cv2
import numpy as np
import tensorflow as tf
import datetime
import csv
import os

# Define Constants
CLASS_NAMES = ["Clear Skin", "Dark Spots", "Puffy Eyes", "Wrinkles"]
LOG_FILE = "inference_logs.csv"

class DermalScanBackend:
    def __init__(self, model_path, prototxt, weights):
        try:
            # 1. Load Skin Analysis Model
            self.model = tf.keras.models.load_model(model_path, compile=False)
            
            # 2. Load Face Detector (OpenCV SSD)
            self.face_net = cv2.dnn.readNetFromCaffe(prototxt, weights)
            
            self._warmup()
            print("✅ Backend: System initialized successfully (Multi-Face Enabled).")
        except Exception as e:
            print(f"❌ Backend Error: {e}")
            raise RuntimeError(f"Failed to load models: {e}")

    def _warmup(self):
        """Run a dummy prediction to load model into memory."""
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)

    def detect_and_predict(self, img_np, conf_threshold=0.5, enable_logging=True):
        h, w = img_np.shape[:2]
        
        # 1. Face Detection (OpenCV DNN)
        blob = cv2.dnn.blobFromImage(cv2.resize(img_np, (300, 300)), 1.0, 
                                   (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        results = []

        # 2. Loop through ALL detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > conf_threshold:
                # Get Box Coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")

                # Safety Check: Ensure box is within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Skip if box is invalid or too small
                if x2 <= x1 or y2 <= y1: continue

                # 3. Extract Face ROI
                face_roi = img_np[y1:y2, x1:x2]
                if face_roi.size == 0: continue

                # 4. AI Classification
                face_input = cv2.resize(face_roi, (224, 224)).astype("float32") / 255.0
                face_input = np.expand_dims(face_input, axis=0)

                preds = self.model.predict(face_input, verbose=0)[0]
                preds = preds.astype(float)
                
                sorted_idx = np.argsort(preds)[::-1]
                top1 = sorted_idx[0]
                
                label = CLASS_NAMES[top1]
                conf_score = preds[top1] * 100

                # 5. Texture Analysis
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
                variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                smoothness = max(0, min(100, 100 - (variance / 15)))

                # Store Result for this face
                face_data = {
                    "box": (x1, y1, x2, y2),
                    "label": label,
                    "confidence": round(conf_score, 2),
                    "smoothness": round(smoothness, 2),
                    "variance": round(variance, 2),
                    "probabilities": {CLASS_NAMES[k]: round(preds[k], 4) for k in range(len(CLASS_NAMES))}
                }
                
                results.append(face_data)

                if enable_logging:
                    self.log_prediction(label, conf_score, smoothness)

        return {
            "found": len(results) > 0,
            "faces": results  # Return list of all faces
        }

    def log_prediction(self, label, conf, smooth):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Prediction", "Confidence", "Smoothness"])
        
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, label, f"{conf:.2f}", f"{smooth:.2f}"])

    def get_clinical_advice(self, condition):
        advice = {
            "Clear Skin": ["✅ Maintain hydration", "🛡️ Apply SPF 50 regularly", "💧 Use Hyaluronic Acid"],
            "Dark Spots": ["🧪 Vitamin C Serum (AM)", "🌙 Niacinamide (PM)", "🚫 Avoid direct sun"],
            "Puffy Eyes": ["🧊 Cold compress", "😴 Check sleep schedule", "🧴 Caffeine eye cream"],
            "Wrinkles": ["🧬 Retinol treatment", "🧱 Ceramide moisturizer", "⚡ Peptide serums"]
        }
        return advice.get(condition, ["Consult a dermatologist"])
