import sys
print(f"Python is running from: {sys.executable}")

try:
    import mediapipe
    print(f"✅ MediaPipe found at: {mediapipe.__file__}")
    
    import mediapipe.python.solutions.face_detection
    print("✅ Solutions submodule loaded successfully!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
except AttributeError as e:
    print(f"❌ Attribute Error (Corrupted): {e}")
    print("This means the folder exists but is empty or broken.")