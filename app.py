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

# ---------------- CUSTOM CSS (ROYAL BLUE THEME + FIXES) ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    /* 1. GLOBAL BACKGROUND - ROYAL BLUE THEME */
    .stApp {
        background: linear-gradient(135deg, #000428 0%, #004e92 100%);
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
    }
    
    /* 2. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #000428;
        border-right: 2px solid #00d2ff;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* 3. GLOWING HEADER */
    @keyframes blink {
        0%, 100% { text-shadow: 0 0 10px #00d2ff, 0 0 20px #00d2ff; opacity: 1; }
        50% { text-shadow: none; opacity: 0.8; }
    }
    
    .main-title {
        font-size: 75px;
        font-weight: 900;
        text-align: center;
        color: #ffffff;
        animation: blink 2s infinite alternate;
        margin-bottom: 5px;
        letter-spacing: 5px;
        text-transform: uppercase;
    }
    
    .sub-title {
        font-size: 22px;
        color: #00d2ff !important;
        text-align: center;
        margin-bottom: 40px;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.5);
    }

    /* 4. GLASS CARDS */
    .glass-card {
        background: rgba(0, 4, 40, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* 5. VISIBILITY FIX: FILE UPLOADER & CAMERA */
    /* Container Styling */
    [data-testid='stFileUploader'], [data-testid='stCameraInput'] {
        background-color: rgba(255, 255, 255, 0.95); /* White background for contrast */
        border: 2px dashed #00d2ff;
        border-radius: 15px;
        padding: 20px;
    }
    /* Text Inside Uploader - Force Black */
    [data-testid='stFileUploader'] section > div,
    [data-testid='stFileUploader'] label,
    [data-testid='stFileUploader'] span,
    [data-testid='stFileUploader'] small,
    [data-testid='stCameraInput'] label,
    [data-testid='stCameraInput'] span {
        color: #000000 !important;
    }
    /* Buttons Inside */
    [data-testid='stFileUploader'] button {
        background-color: #00d2ff !important;
        color: #000428 !important;
        border: none !important;
        font-weight: bold !important;
    }
    /* Camera Button specifically */
    [data-testid='stCameraInput'] button {
        background-color: #ff00de !important;
        color: white !important;
        border: none !important;
    }

    /* 6. GLOBAL ACTION BUTTONS */
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 15px 30px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 25px rgba(0, 210, 255, 0.8);
    }

    /* 7. METRICS & TEXT */
    .metric-value { 
        font-size: 50px; 
        font-weight: 900; 
        color: #00d2ff; 
        text-shadow: 0 0 20px #00d2ff;
    }
    .metric-label {
        font-size: 16px;
        color: #ffffff;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .diagnosis-title {
        font-size: 42px;
        font-weight: 800;
        color: #ff00de !important;
        text-shadow: 0 0 20px #ff00de;
        margin-bottom: 20px;
    }

    .rec-item {
        background: rgba(0, 210, 255, 0.1);
        border-left: 5px solid #00d2ff;
        color: white !important;
        padding: 15px;
        margin-bottom: 12px;
        font-size: 18px;
        border-radius: 0 10px 10px 0;
    }

    /* 8. NEW FEATURE: BREAKDOWN BARS */
    .bar-container {
        margin-bottom: 12px;
    }
    .bar-label {
        display: flex;
        justify-content: space-between;
        color: #e0e0e0;
        font-size: 14px;
        margin-bottom: 5px;
        font-weight: 600;
    }
    .bar-bg {
        background: rgba(255,255,255,0.1);
        height: 10px;
        border-radius: 5px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #00d2ff, #ff00de);
        border-radius: 5px;
    }

    /* HIDE FOOTER */
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
    # Dark Mode Plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    # Transparent Background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    data = np.array(list(probs.values())).reshape(1, -1)
    
    # VISIBILITY FIX: Using 'mako' (Deep Blue/Green) - Great contrast with Cyan/White text
    sns.heatmap(data, annot=True, annot_kws={"size": 18, "weight": "bold", "color": "#00d2ff"}, 
                xticklabels=probs.keys(), yticklabels=[""], cmap="mako", ax=ax, cbar=False)
    
    ax.tick_params(axis='x', colors='white', labelsize=14)
    ax.set_yticks([]) 
    
    # Add border to plot
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('#00d2ff')
        
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

# ---------------- LIVE PROCESSOR ----------------
class LiveProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        res = backend.detect_and_predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), enable_logging=False)
        
        if res["found"]:
            x1, y1, x2, y2 = res["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            label = f"{res['label']} ({int(res['confidence']) }%)"
            cv2.putText(img, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return img

# ---------------- UI HEADER ----------------
st.markdown('<div class="main-title">DERMALSCAN</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced AI Skin Analysis Interface</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.markdown("### ⚙️ System Config")
    mode = st.radio("Input Source", ["📁 Upload Image", "📷 Capture Image", "🎥 Live Stream"])
    st.markdown("---")
    
    if mode in ["📷 Capture Image", "🎥 Live Stream"]:
        camera_type = st.radio("Camera Device", ["Built-in Webcam", "IP Camera (Phone)"])
        ip_cam_url = ""
        if camera_type == "IP Camera (Phone)":
            st.info("ℹ️ Enter URL from IP Webcam app.")
            ip_cam_url = st.text_input("Stream URL", "http://192.168.1.5:8080/video")
            
    st.markdown("---")
    thresh = st.slider("AI Sensitivity", 0.0, 1.0, 0.6)

# ---------------- MAIN LOGIC ----------------
img_np = None

if mode == "📁 Upload Image":
    f = st.file_uploader("Upload Image File", ["jpg", "png", "jpeg"])
    if f: img_np = np.array(Image.open(f).convert("RGB"))

elif mode == "📷 Capture Image":
    if camera_type == "Built-in Webcam":
        c = st.camera_input("Take Photo")
        if c: img_np = np.array(Image.open(c).convert("RGB"))
    else:
        st.markdown("#### 📱 Remote Camera")
        if ip_cam_url:
            if st.button("📸 Snap Photo"):
                try:
                    cap = cv2.VideoCapture(ip_cam_url)
                    ret, frame = cap.read()
                    if ret:
                        img_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.success("Captured!")
                    cap.release()
                except:
                    st.error("Connection failed.")

elif mode == "🎥 Live Stream":
    st.markdown("### 🔴 Real-Time Feed")
    if camera_type == "Built-in Webcam":
        rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(key="live", video_transformer_factory=LiveProcessor, rtc_configuration=rtc_configuration)
    else:
        st.info(f"Streaming from {ip_cam_url}...")
        video_spot = st.empty()
        stop = st.button("Stop Stream")
        if ip_cam_url and not stop:
            cap = cv2.VideoCapture(ip_cam_url)
            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret: break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = backend.detect_and_predict(rgb_frame, thresh, enable_logging=False)
                if res["found"]:
                    x1, y1, x2, y2 = res["box"]
                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    label = f"{res['label']} ({int(res['confidence']) }%)"
                    cv2.putText(rgb_frame, label, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                video_spot.image(rgb_frame, channels="RGB")
            cap.release()

# ---------------- RESULTS ----------------
if img_np is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        analyze = st.button("🔍 INITIALIZE SCAN", type="primary", use_container_width=True)

    if analyze:
        with st.spinner("Processing biometric data..."):
            time.sleep(0.5)
            res = backend.detect_and_predict(img_np, thresh)
            
            if res["found"]:
                x1,y1,x2,y2 = res["box"]
                cv2.rectangle(img_np, (x1,y1), (x2,y2), (0, 210, 255), 5)
                
                # Metric: Face Coverage %
                h, w, _ = img_np.shape
                face_area = (x2-x1) * (y2-y1)
                total_area = h * w
                face_coverage = (face_area / total_area) * 100
                
                # ROW 1: Visuals & Logs
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.image(img_np, caption="Identified Region", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    logs = generate_logs(res)
                    log_html = '<div style="background:rgba(0,0,0,0.5); padding:15px; border-radius:10px; height:200px; overflow-y:auto; color:#00ff00; font-family:monospace; border:1px solid #00d2ff;">' + "".join([f"<div>> {l}</div>" for l in logs]) + '</div>'
                    st.markdown(log_html, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    # METRICS
                    m1, m2, m3 = st.columns(3)
                    m1.markdown(f'<div align="center"><div class="metric-value">{res["confidence"]:.0f}%</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
                    m2.markdown(f'<div align="center"><div class="metric-value">{res["smoothness"]:.0f}</div><div class="metric-label">Texture</div></div>', unsafe_allow_html=True)
                    m3.markdown(f'<div align="center"><div class="metric-value">{face_coverage:.0f}%</div><div class="metric-label">Face Area</div></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # ROW 2: Clinical Data
                st.markdown("### 📋 Analysis Report")
                cl1, cl2 = st.columns([1, 1])
                with cl1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="diagnosis-title">{res["label"]}</div>', unsafe_allow_html=True)
                    tips = backend.get_clinical_advice(res["label"])
                    for t in tips:
                        clean_t = t.replace("**", "<b>").replace(":", ":</b>")
                        st.markdown(f'<div class="rec-item">{clean_t}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with cl2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### 🧪 Probability Heatmap")
                    st.pyplot(plot_heatmap(res["probabilities"]))
                    
                    # NEW FEATURE: Detailed Condition Breakdown
                    st.markdown("<br>#### 📊 Detailed Condition Breakdown", unsafe_allow_html=True)
                    for key, val in res["probabilities"].items():
                        width = float(val) * 100
                        st.markdown(f"""
                        <div class="bar-container">
                            <div class="bar-label">
                                <span>{key}</span>
                                <span>{width:.1f}%</span>
                            </div>
                            <div class="bar-bg">
                                <div class="bar-fill" style="width: {width}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown('</div>', unsafe_allow_html=True)

                # ROW 3: Exports
                st.markdown("### 📂 Export Data")
                e1, e2, e3 = st.columns(3)
                buf = io.BytesIO()
                Image.fromarray(img_np).save(buf, format="JPEG")
                e1.download_button("🖼️ Save Image", buf.getvalue(), "scan.jpg", "image/jpeg", use_container_width=True)
                pdf_data = generate_pdf(res["label"], res["confidence"], res["smoothness"], tips)
                e2.download_button("📄 PDF Report", pdf_data, "report.pdf", "application/pdf", use_container_width=True)
                if os.path.exists("inference_logs.csv"):
                    with open("inference_logs.csv", "rb") as f:
                        e3.download_button("📊 CSV Logs", f, "logs.csv", "text/csv", use_container_width=True)
                        
            else:
                st.warning("⚠️ No face detected in the image.")
