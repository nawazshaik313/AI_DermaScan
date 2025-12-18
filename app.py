import streamlit as st
import numpy as np
from PIL import Image
import time
import cv2 
import datetime
import io
import os
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Import Backend
from backend import DermalScanBackend

# ---------------- CONFIGURATION ----------------
st.set_page_config(page_title="DermalScan Elite", page_icon="💠", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    .stApp { background-color: #F0F2F6; font-family: 'Roboto', sans-serif; }
    
    .main-header { 
        background: linear-gradient(90deg, #009688 0%, #263238 100%); 
        padding: 30px; border-radius: 20px; color: white; 
        text-align: center; margin-bottom: 40px; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.15); 
    }
    
    .glass-card { 
        background: white; padding: 30px; border-radius: 20px; 
        box-shadow: 0 10px 40px rgba(0,0,0,0.08); margin-bottom: 30px;
    }
    
    /* BIGGER TEXT STYLES */
    .diagnosis-title { 
        font-size: 48px !important; font-weight: 900; color: #2c3e50; 
        border-bottom: 5px solid #009688; display: inline-block; 
        padding-bottom: 10px; margin-bottom: 20px;
    }
    
    .rec-item {
        font-size: 24px !important; color: #34495e; margin-bottom: 18px;
        padding: 15px; background-color: #f8f9fa; border-radius: 12px;
        border-left: 8px solid #009688; line-height: 1.5;
    }

    .metric-value { font-size: 42px; font-weight: 900; color: #009688; }
    .console-box { 
        background-color: #1E1E1E; color: #00FF00; padding: 20px; 
        border-radius: 12px; height: 250px; overflow-y: auto; 
        font-family: monospace; font-size: 14px; 
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- INITIALIZE BACKEND ----------------
@st.cache_resource
def get_backend():
    return DermalScanBackend(
        "skin_condition_model_m4_optimized.keras",
        "face_detector/deploy.prototxt",
        "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    )

try:
    backend = get_backend()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# ---------------- HELPERS ----------------
def sanitize(text):
    return text.encode("latin-1", "ignore").decode("latin-1")

def generate_pdf(label, conf, smooth, advice):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "DermalScan Medical Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(0, 8, f"Condition: {sanitize(label)}", ln=True)
    pdf.cell(0, 8, f"Confidence: {conf:.2f}%", ln=True)
    pdf.cell(0, 8, f"Smoothness: {smooth:.2f}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Recommendations", ln=True)
    pdf.set_font("Arial", size=11)
    for tip in advice:
        pdf.multi_cell(0, 8, f"- {sanitize(tip)}")
    return pdf.output(dest="S").encode("latin-1")

def plot_heatmap(probs):
    fig, ax = plt.subplots(figsize=(10, 4)) 
    data = np.array(list(probs.values())).reshape(1, -1)
    # Using 'GnBu' for Teal gradient
    sns.heatmap(data, annot=True, annot_kws={"size": 14, "weight": "bold"}, 
                xticklabels=probs.keys(), yticklabels=["Conf"], cmap="GnBu", ax=ax, cbar=False)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    return fig

def generate_logs(res):
    t = datetime.datetime.now().strftime("%H:%M:%S")
    return [
        f"[{t}] SYSTEM: Connected",
        f"[{t}] DETECT: Box {res['box']}",
        f"[{t}] MATH: Var {res['variance']:.2f}",
        f"[{t}] AI: {res['label']} ({res['confidence']}%)",
        f"[{t}] DB: Log Saved to CSV"
    ]

# ---------------- LIVE STREAM PROCESSOR (LOCAL) ----------------
class LiveProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Run inference without logging
        res = backend.detect_and_predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), enable_logging=False)
        
        if res["found"]:
            x1, y1, x2, y2 = res["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{res['label']} ({int(res['confidence']) }%)"
            cv2.putText(img, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        return img

# ---------------- UI LAYOUT ----------------
st.markdown('<div class="main-header"><h1>DermalScan Elite</h1><p style="font-size: 20px;">AI-Powered Dermatological Diagnostics System</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=120)
    st.markdown("### ⚙️ Control Panel")
    
    # INPUT MODE SELECTOR
    mode = st.radio("Input Source", ["📁 Upload Image", "📷 Capture Image", "🎥 Live Stream"])
    
    st.markdown("---")
    
    # CAMERA DEVICE SELECTOR
    if mode in ["📷 Capture Image", "🎥 Live Stream"]:
        st.markdown("### 📡 Camera Device")
        camera_type = st.radio("Choose Camera:", ["Built-in / USB Webcam", "IP Camera (Phone)"])
        
        ip_cam_url = ""
        if camera_type == "IP Camera (Phone)":
            st.info("ℹ️ Download an 'IP Webcam' app on your phone and enter the video URL here.")
            ip_cam_url = st.text_input("IP Camera URL", "http://192.168.31.128:8080/video")
            
    st.markdown("---")
    thresh = st.slider("Sensitivity", 0.0, 1.0, 0.6)

# ---------------- LOGIC ----------------
img_np = None

# --- MODE 1: FILE UPLOAD ---
if mode == "📁 Upload Image":
    f = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])
    if f: img_np = np.array(Image.open(f).convert("RGB"))

# --- MODE 2: CAPTURE IMAGE ---
elif mode == "📷 Capture Image":
    if camera_type == "Built-in / USB Webcam":
        c = st.camera_input("Take Photo")
        if c: img_np = np.array(Image.open(c).convert("RGB"))
        
    else: # IP Camera Mode
        st.markdown("#### 📱 Remote Phone Camera Capture")
        if ip_cam_url:
            if st.button("📸 Snap Photo from Phone"):
                try:
                    cap = cv2.VideoCapture(ip_cam_url)
                    ret, frame = cap.read()
                    if ret:
                        img_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.success("Image captured successfully!")
                    else:
                        st.error("Failed to connect to phone. Check URL and Wi-Fi.")
                    cap.release()
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# --- MODE 3: LIVE STREAM ---
elif mode == "🎥 Live Stream":
    st.markdown("### 🔴 Real-Time Analysis")
    
    if camera_type == "Built-in / USB Webcam":
        # Standard WebRTC for Local Webcam
        rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(
            key="live", 
            video_transformer_factory=LiveProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False}
        )
        st.info("Using Local Device Camera")
        
    else: # IP Camera Stream
        st.info(f"Connecting to Phone: {ip_cam_url} ...")
        
        # Placeholder for video
        video_spot = st.empty()
        stop_btn = st.button("Stop Stream")
        
        if ip_cam_url and not stop_btn:
            cap = cv2.VideoCapture(ip_cam_url)
            
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Video stream unavailable.")
                    break
                
                # Process Frame (Backend)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = backend.detect_and_predict(rgb_frame, thresh, enable_logging=False)
                
                # Draw Box
                if res["found"]:
                    x1, y1, x2, y2 = res["box"]
                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    label = f"{res['label']} ({int(res['confidence']) }%)"
                    cv2.putText(rgb_frame, label, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
                # Show in Streamlit
                video_spot.image(rgb_frame, channels="RGB")
                
            cap.release()

# ---------------- RESULTS PROCESSING (STATIC ONLY) ----------------
# Logic for processing captured/uploaded images
if img_np is not None:
    if st.button("🚀 INITIALIZE SCAN", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            time.sleep(0.5)
            res = backend.detect_and_predict(img_np, thresh)
            
            if res["found"]:
                # Draw Box
                x1,y1,x2,y2 = res["box"]
                cv2.rectangle(img_np, (x1,y1), (x2,y2), (0,150,136), 4)
                
                # --- ROW 1: LARGE IMAGE & METRICS ---
                c1, c2 = st.columns([1.2, 1]) 
                with c1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.image(img_np, caption="Analyzed Region", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with c2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    logs = generate_logs(res)
                    log_html = '<div class="console-box">' + "".join([f"<div>> {l}</div>" for l in logs]) + '</div>'
                    st.markdown(log_html, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.markdown(f'<div align="center"><div class="metric-value">{res["confidence"]:.0f}%</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
                    m2.markdown(f'<div align="center"><div class="metric-value">{res["smoothness"]:.1f}</div><div class="metric-label">Smoothness</div></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # --- ROW 2: CLINICAL REPORT ---
                st.markdown("### 📋 Clinical Report")
                r1, r2 = st.columns([1, 1]) 
                
                with r1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="diagnosis-title">{res["label"]}</div>', unsafe_allow_html=True)
                    tips = backend.get_clinical_advice(res["label"])
                    for t in tips:
                        clean_t = t.replace("**", "<b>").replace(":", ":</b>")
                        st.markdown(f'<div class="rec-item">{clean_t}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with r2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### 🧪 Probability Heatmap")
                    st.pyplot(plot_heatmap(res["probabilities"]))
                    st.markdown('</div>', unsafe_allow_html=True)

                # --- EXPORTS ---
                st.markdown("### 📂 Exports")
                e1, e2, e3 = st.columns(3)
                
                buf = io.BytesIO()
                Image.fromarray(img_np).save(buf, format="JPEG")
                e1.download_button("🖼️ Download Image", buf.getvalue(), "scan.jpg", "image/jpeg", use_container_width=True)
                
                pdf_data = generate_pdf(res["label"], res["confidence"], res["smoothness"], tips)
                e2.download_button("📄 Download Report", pdf_data, "report.pdf", "application/pdf", use_container_width=True)
                
                if os.path.exists("inference_logs.csv"):
                    with open("inference_logs.csv", "rb") as f:
                        e3.download_button("📊 Download Logs", f, "logs.csv", "text/csv", use_container_width=True)
                        
            else:
                st.warning("⚠️ No face detected. Please try a different angle.")