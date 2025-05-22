import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import io
import pandas as pd

st.set_page_config(page_title="DeepCheck Visual", layout="wide")

# Constants
IMG_SIZE = (224, 224)
CLASS_MAP = {0: 'Fake', 1: 'Real'}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Assets\Test")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Assets\Model\mobilenetv2.h5")

model = load_model()

# Load test images from dataset
def list_test_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Preprocessing function
def preprocess_image(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    return tf.expand_dims(img_tensor, axis=0)

# Annotate image with label and confidence
def draw_prediction_label(img, label, confidence):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"{label} ({confidence:.2%})"
    draw.text((10, 10), text, fill="red", font=font)
    return img

# --- UI START ---
st.title("🧠 DeepCheck Visual")
st.subheader("Fake vs Real Image Classifier")

with st.expander("ℹ️ About this app"):
    st.write(
        """
        Welcome to **DeepCheck Visual**!  
        This app leverages a **MobileNetV2** deep learning model to classify images as **Fake** or **Real**.  

        Simply upload one or more JPG or PNG images, and the app will display:
        - Predicted label (Fake or Real)  
        - Confidence score for each prediction  
        - Detailed probability distribution across classes  

        You can also test images from the built-in dataset and download annotated images or classification reports.

        *Note: The model was trained on a specific dataset — results may vary on other image types.*
        """
    )


# Tabs for manual upload vs test dataset
tab1, tab2 = st.tabs(["📤 Upload Image", "🧪 Test from Dataset"])

with tab1:
    uploaded_files = st.file_uploader(
        "Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    
    if uploaded_files:
        col1, col2, col3 = st.columns([1, 2, 2])

        # Prepare list to collect report data
        report_data = []

        with col1:
            # List filenames in a selectbox to preview one at a time
            filenames = [f.name for f in uploaded_files]
            selected_filename = st.selectbox("Select an image to preview", filenames)
            selected_file = next(f for f in uploaded_files if f.name == selected_filename)
            img_preview = Image.open(selected_file)

            st.image(img_preview, caption=selected_filename, use_container_width=True)

        with col2:
            # Predict on selected image
            img_tensor = preprocess_image(img_preview)
            prediction = model.predict(img_tensor)
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))

            st.markdown(f"### Prediction: **{CLASS_MAP[predicted_class]}**")
            st.markdown(f"**Confidence:** {confidence:.2%}")
            st.progress(min(confidence, 1.0))

            st.markdown("#### All Class Probabilities")
            for idx, prob in enumerate(prediction[0]):
                st.write(f"{CLASS_MAP[idx]}: {prob:.2%}")
                st.progress(float(prob))

            # Download labeled image button
            labeled_img = draw_prediction_label(img_preview.copy(), CLASS_MAP[predicted_class], confidence)
            buf = io.BytesIO()
            labeled_img.save(buf, format="PNG")
            st.download_button("📥 Download Labeled Image", buf.getvalue(), file_name=f"{selected_filename}_prediction.png")

        with col3:
            st.markdown("### Classification Report")
            report_data = []
            for f in uploaded_files:
                img = Image.open(f)
                img_tensor = preprocess_image(img)
                pred = model.predict(img_tensor)
                cls = np.argmax(pred)
                conf = float(np.max(pred))
                report_data.append({
                    "Filename": f.name,
                    "Classification": CLASS_MAP[cls],
                    "Confidence": f"{conf:.2%}"
                })

            df_report = pd.DataFrame(report_data)
            
            # Multi-select to delete rows
            to_delete = st.multiselect(
                "Select files to delete from report",
                options=df_report["Filename"].tolist()
            )

            st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #e74c3c;
                color: white;
            }
            div.stButton > button:first-child:hover {
                background-color: #c0392b;
                color: white;
            }
            </style>""", unsafe_allow_html=True)

            if st.button("Delete Selected"):
                if to_delete:
                    df_report = df_report[~df_report["Filename"].isin(to_delete)].reset_index(drop=True)
                    st.success(f"Deleted {len(to_delete)} entries.")
                else:
                    st.warning("No files selected to delete.")

            # Show updated report
            st.table(df_report)
            
            # Prepare Excel download
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                df_report.to_excel(writer, index=False, sheet_name='Report')
            towrite.seek(0)

            st.download_button(
                label="📥 Download Report as Excel",
                data=towrite,
                file_name="classification_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("Upload one or more images to classify.")
        
with tab2:
    test_images = list_test_images(DATASET_PATH)
    if test_images:
        col1, col2, col3 = st.columns([1, 2, 2])  # 3 columns with ratio 1:2:2

        with col1:
            selected_image = st.selectbox("Choose test image", test_images)
            img_path = os.path.join(DATASET_PATH, selected_image)

        with col2:
            img = Image.open(img_path)
            st.image(img, caption=selected_image, use_container_width=True)

        with col3:
            img_tensor = preprocess_image(img)
            prediction = model.predict(img_tensor)
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))

            st.markdown(f"### Prediction: **{CLASS_MAP[predicted_class]}**")
            st.markdown(f"**Confidence:** {confidence:.2%}")
            st.progress(min(confidence, 1.0))

            st.markdown("#### All Class Probabilities")
            for idx, prob in enumerate(prediction[0]):
                st.write(f"{CLASS_MAP[idx]}: {prob:.2%}")
                st.progress(float(prob))
    else:
        st.warning("No images found in `.Test` directory.")
