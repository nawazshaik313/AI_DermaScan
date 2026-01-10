import cv2
import threading
import time
from ultralytics import YOLO

# ==========================================
# 1. THREADED CAMERA CLASS
# ==========================================
class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.capture.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.capture.isOpened():
                self.ret, self.frame = self.capture.read()
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.capture.release()

# ==========================================
# 2. MAIN DETECTION LOOP
# ==========================================
def run_detection():
    print("Loading Model...")
    model = YOLO("yolov8n.pt") 
    
    cap = ThreadedCamera(0)
    time.sleep(2) # Allow camera to warm up
    
    frame_count = 0
    skip_frames = 3
    stored_boxes = [] 
    
    print("Starting Stream. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Resize
        small_frame = cv2.resize(frame, (640, 480))

        # 2. Detect (Every 3rd frame)
        if frame_count % skip_frames == 0:
            results = model.predict(small_frame, verbose=False, conf=0.5)
            stored_boxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Store box + label
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]
                    stored_boxes.append((x1, y1, x2, y2, label))

        # 3. Draw
        for (x1, y1, x2, y2, label) in stored_boxes:
            cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(small_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Display (Standard OpenCV Window)
        cv2.imshow("Fast Detection", small_frame)
        
        frame_count += 1

        # 5. Exit Logic
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()