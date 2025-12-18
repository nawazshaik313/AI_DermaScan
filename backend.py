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
        """Initialize the AI models."""
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.face_net = cv2.dnn.readNetFromCaffe(prototxt, weights)
            self._warmup()
            print("✅ Backend: System initialized successfully.")
        except Exception as e:
            print(f"❌ Backend Error: {e}")
            raise RuntimeError(f"Failed to load models: {e}")

    def _warmup(self):
        """Run a dummy prediction to load model into memory."""
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        self.model.predict(dummy, verbose=0)

    def detect_and_predict(self, img_np, conf_threshold=0.6, enable_logging=True):
        """
        Main Pipeline: Face Detection -> Preprocessing -> Inference -> Logging
        """
        h, w = img_np.shape[:2]
        
        # 1. Face Detection
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img_np, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        best_idx = -1
        max_conf = 0

        # Find the most confident face
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > conf_threshold and conf > max_conf:
                max_conf = conf
                best_idx = i

        if best_idx < 0:
            return {"found": False}

        # 2. ROI Extraction
        box = detections[0, 0, best_idx, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        face_roi = img_np[y1:y2, x1:x2]
        if face_roi.size == 0:
            return {"found": False}

        # 3. AI Classification
        face_input = cv2.resize(face_roi, (224, 224)).astype("float32") / 255.0
        face_input = np.expand_dims(face_input, axis=0)

        preds = self.model.predict(face_input, verbose=0)[0]
        preds = preds.astype(float)
        
        # Logic for primary class
        sorted_idx = np.argsort(preds)[::-1]
        top1 = sorted_idx[0]
        
        label = CLASS_NAMES[top1]
        confidence = preds[top1] * 100

        # 4. Texture Analysis (Math)
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        smoothness = max(0, min(100, 100 - (variance / 15)))

        # 5. Logging (Optional for Live Video)
        if enable_logging:
            self.log_prediction(label, confidence, smoothness)

        return {
            "found": True,
            "box": (x1, y1, x2, y2),
            "label": label,
            "confidence": round(confidence, 2),
            "smoothness": round(smoothness, 2),
            "variance": round(variance, 2),
            "probabilities": {CLASS_NAMES[i]: round(preds[i], 4) for i in range(len(CLASS_NAMES))}
        }

    def log_prediction(self, label, conf, smooth):
        """Saves scan data to CSV."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Prediction", "Confidence", "Smoothness"])
        
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, label, f"{conf:.2f}", f"{smooth:.2f}"])

    def get_clinical_advice(self, condition):
        """Returns medical advice based on detection."""
        advice = {
            "Clear Skin": ["✅ Maintain hydration", "🛡️ Apply SPF 50 regularly", "💧 Use Hyaluronic Acid"],
            "Dark Spots": ["🧪 Vitamin C Serum (AM)", "🌙 Niacinamide (PM)", "🚫 Avoid direct sun"],
            "Puffy Eyes": ["🧊 Cold compress", "😴 Check sleep schedule", "🧴 Caffeine eye cream"],
            "Wrinkles": ["🧬 Retinol treatment", "🧱 Ceramide moisturizer", "⚡ Peptide serums"]
        }
        return advice.get(condition, ["Consult a dermatologist"])