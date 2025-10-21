# updated_app.py
import os
import warnings
import platform
import time

# ------------------ Quiet noisy TensorFlow / Keras logs ------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # only show errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disable oneDNN op messages (optional)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------ Standard imports ------------------
import streamlit as st
import numpy as np
import cv2
import cvlib as cv
from deepface import DeepFace
from PIL import Image, ImageDraw
import json
from datetime import datetime
import pandas as pd
from streamlit_lottie import st_lottie

# ------------------ Config & folders ------------------
SAVE_DIR = "detected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

st.set_page_config(layout='wide', page_title="Face Vision", page_icon="👦")

# ------------------ Helpers ------------------
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def deepface_analyze_safe(img_array, actions=['emotion'], detector_backend='opencv', enforce_detection=False):
    """
    Call DeepFace.analyze while handling return types and errors.
    Accepts an image as a numpy array in BGR format (as OpenCV usually uses).
    Returns: (dominant_emotion_str, emotions_dict) or (None, None) on failure
    """
    try:
        result = DeepFace.analyze(img_array, actions=actions,
                                  detector_backend=detector_backend,
                                  enforce_detection=enforce_detection)
    except Exception as e:
        # log error (do not crash)
        st.debug(f"DeepFace analyze error: {e}")
        return None, None

    # DeepFace can return either a dict or a list with a single dict
    if isinstance(result, list) and len(result) > 0:
        result = result[0]
    if not isinstance(result, dict):
        return None, None

    dominant = result.get('dominant_emotion') or result.get('dominant', None)
    emotions = result.get('emotion') or {}
    return dominant, emotions

# ------------------ UI: Lottie & Sidebar ------------------
# Place your animation.json file path (must exist)
LOTTIE_FILE = "animation.json"
logo = None
try:
    logo = load_lottiefile(LOTTIE_FILE)
except Exception:
    logo = None

with st.sidebar:
    st.title('Experience the power of Computer Vision')
    st.success('Detect emotions in real-time using your webcam or an image.')
    mode = st.selectbox('Select Mode', ['<Select>', 'Capture', 'Webcam'])

st.markdown("<h1 style='text-align: center;'>Face Vision</h1><br>", unsafe_allow_html=True)
if logo:
    st_lottie(logo, speed=1, width=250, height=250)

if mode == '<Select>':
    st.warning('👈 Select a mode from the sidebar to start!')

# ------------------ Persistent emotion log in session state ------------------
if 'emotion_log' not in st.session_state:
    st.session_state['emotion_log'] = []  # each entry: dict with Time, Dominant, Angry, Happy, ...

# ------------------ CAPTURE MODE ------------------
if mode == 'Capture':
    image = st.camera_input("Capture a snapshot")
    if image is not None:
        pil_img = Image.open(image).convert("RGB")
        image_np = np.array(pil_img)  # RGB
        # Convert to BGR for OpenCV/DeepFace consistency
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        faces, confidences = cv.detect_face(image_bgr)
        draw = ImageDraw.Draw(pil_img)

        last_label = "No face detected"
        for f in faces:
            startX, startY, endX, endY = f
            # clamp coords
            h, w = image_bgr.shape[:2]
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            face_crop_bgr = image_bgr[startY:endY, startX:endX]
            if face_crop_bgr.size == 0:
                continue

            # Resize to speed up
            try:
                face_crop_bgr_small = cv2.resize(face_crop_bgr, (224, 224))
            except Exception:
                face_crop_bgr_small = face_crop_bgr

            dominant, emotions = deepface_analyze_safe(face_crop_bgr_small,
                                                       actions=['emotion'],
                                                       detector_backend='opencv',
                                                       enforce_detection=False)
            if dominant is None:
                label = "Unknown"
            else:
                label = dominant
            last_label = label

            # Save log (store all emotion probs; convert floats to rounded numbers)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {"Time": timestamp, "Dominant": label}
            # convert emotions dict to rounded floats
            for k, v in (emotions or {}).items():
                try:
                    log_entry[k.capitalize()] = round(float(v), 2)
                except Exception:
                    log_entry[k.capitalize()] = v
            st.session_state['emotion_log'].append(log_entry)

            # Draw rectangle and label onto PIL image (RGB)
            draw.rectangle(((startX, startY), (endX, endY)), outline="green", width=2)
            draw.text((startX, startY - 12), label, fill="green")

        # Save captured image
        tsfile = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = os.path.join(SAVE_DIR, f"captured_{tsfile}.png")
        pil_img.save(img_filename)

        st.image(pil_img, caption=f"Processed Image (Saved as {img_filename})", use_column_width=True)
        st.write('**Emotion Detected:**', last_label)

# ------------------ WEBCAM MODE ------------------
if mode == 'Webcam':
    st.write("### Real-Time Emotion Detection")

    # Buttons to control streaming (use session_state flags)
    if 'webcam_running' not in st.session_state:
        st.session_state['webcam_running'] = False

    start_col, stop_col = st.columns([1, 1])
    with start_col:
        start_btn = st.button("Start Webcam")
    with stop_col:
        stop_btn = st.button("Stop Webcam")

    # Start/stop behavior
    if start_btn:
        st.session_state['webcam_running'] = True
    if stop_btn:
        st.session_state['webcam_running'] = False

    # Only open camera when flag is True
    if st.session_state['webcam_running']:
        # OpenCV backend selection for Windows to avoid MSMF warnings
        if platform.system().lower() == "windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Failed to open webcam. Check permissions and device.")
        else:
            frame_placeholder = st.empty()
            # prepare video writer (optional)
            tsfile = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = os.path.join(SAVE_DIR, f"recorded_{tsfile}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_filename, fourcc, 15.0, (640, 480))

            frame_count = 0
            PROCESS_EVERY_N_FRAMES = 3  # change to 2 or 4 depending on speed/accuracy tradeoff
            try:
                while st.session_state['webcam_running'] and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to read frame from webcam.")
                        break

                    # Resize frame for display and speed
                    display_frame = cv2.resize(frame, (640, 480))
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                    frame_count += 1
                    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                        # Detect faces on the resized frame
                        faces, confidences = cv.detect_face(display_frame)

                        for f in faces:
                            startX, startY, endX, endY = f
                            # clamp
                            h, w = display_frame.shape[:2]
                            startX, startY = max(0, startX), max(0, startY)
                            endX, endY = min(w - 1, endX), min(h - 1, endY)
                            face_crop_bgr = display_frame[startY:endY, startX:endX]
                            if face_crop_bgr.size == 0:
                                continue

                            # smaller copy
                            try:
                                face_crop_bgr_small = cv2.resize(face_crop_bgr, (224, 224))
                            except Exception:
                                face_crop_bgr_small = face_crop_bgr

                            # Analyze (pass BGR image)
                            dominant, emotions = deepface_analyze_safe(face_crop_bgr_small,
                                                                       actions=['emotion'],
                                                                       detector_backend='opencv',
                                                                       enforce_detection=False)
                            label = dominant if dominant else "Unknown"

                            # overlay label and box
                            cv2.rectangle(frame_rgb, (startX, startY), (endX, endY), (0, 255, 0), 2)
                            cv2.putText(frame_rgb, label, (startX, startY - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            # store log
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            log_entry = {"Time": timestamp, "Dominant": label}
                            for k, v in (emotions or {}).items():
                                try:
                                    log_entry[k.capitalize()] = round(float(v), 2)
                                except Exception:
                                    log_entry[k.capitalize()] = v
                            st.session_state['emotion_log'].append(log_entry)

                    # display frame
                    frame_placeholder.image(frame_rgb, channels="RGB")

                    # write frame to video (BGR)
                    out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

                    # small sleep allows Streamlit to catch UI events and prevents 100% CPU loop
                    time.sleep(0.02)

            finally:
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                st.session_state['webcam_running'] = False
                st.success(f"Video saved as {video_filename}")

# ------------------ REPORT GENERATION (presents whenever logs exist) ------------------
if len(st.session_state['emotion_log']) > 0:
    st.subheader("📊 Emotion Report")
    df = pd.DataFrame(st.session_state['emotion_log'])
    st.dataframe(df)

    # Show a quick summary (most frequent dominant emotions)
    try:
        freq = df['Dominant'].value_counts(normalize=True).mul(100).round(1)
        st.write("**Summary:** Most frequent emotions (percentage):")
        st.write(freq.to_frame(name='Percent'))
    except Exception:
        pass

    # Plot simple line of counts over time for a selected emotion
    try:
        import matplotlib.pyplot as plt
        # prepare counts of dominant emotion over time (rolling)
        df_time = df.copy()
        df_time['Time'] = pd.to_datetime(df_time['Time'])
        df_time_sorted = df_time.sort_values('Time').set_index('Time')
        # show counts per minute for top 3 emotions
        top3 = df_time_sorted['Dominant'].value_counts().nlargest(3).index.tolist()
        fig, ax = plt.subplots(figsize=(8, 3))
        for emo in top3:
            series = (df_time_sorted['Dominant'] == emo).resample('30S').sum()
            ax.plot(series.index, series.values, label=emo)
        ax.legend()
        ax.set_title("Emotion occurrences over time (resampled 30s)")
        st.pyplot(fig, bbox_inches='tight')
    except Exception:
        pass

    # CSV Download (always available)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Report (CSV)", data=csv,
                       file_name=f"emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                       mime="text/csv")

    # Try Excel export; if openpyxl not installed, show a friendly message and skip Excel
    try:
        excel_file = f"emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(excel_file, index=False)
        with open(excel_file, "rb") as f:
            st.download_button("⬇️ Download Report (Excel)", data=f,
                               file_name=excel_file,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except ModuleNotFoundError:
        st.info("Excel export requires 'openpyxl' package. Use CSV download, or install via `pip install openpyxl` to enable Excel export.")
    except Exception as e:
        st.error(f"Excel export failed: {e}")

# ------------------ End ------------------
