# 🧠 DeepCheck Visual

**DeepCheck Visual** is a web-based tool built with **Streamlit** that utilizes a deep learning model (MobileNetV2) to classify images as either **Fake** or **Real**. It provides a simple and interactive interface for both casual users and researchers to analyze image authenticity through deep learning.

## 🚀 Features

- 📤 Upload and classify your own images (JPG, PNG)
- 🧪 Test using a built-in test dataset
- 🖼️ View image previews, prediction results, and confidence scores
- 📈 Visual breakdown of class probabilities
- 🧾 Generate and download an Excel report
- 🧹 Delete selected entries from the report
- 📝 Download annotated prediction images with label & confidence

## 🧠 Model

The app uses a **MobileNetV2** model trained to detect and classify images as:
- **Fake**
- **Real**

The model was trained using TensorFlow and expects RGB images resized to 224x224 pixels.

## 📁 Directory Structure
```
Deepcheck-Visual/
├── Assets/
│   ├── Model/
│   │   └── mobilenetv2.h5          # Trained MobileNetV2 model
│   └── Test/                       # Test images for evaluation
├── app.py                          # Main Streamlit app
└── README.md                       # Project description
```

## 🚀 Getting Started

Follow these steps to set up and run the DeepCheck Visual application locally:

### 1. Clone the Repository

```bash
git clone https://github.com/ngurah909/Deepcheck-Visual.git
cd Deepcheck-Visual
```

### 2. Install Dependencies

Make sure you have Python 3.7+ installed. Then install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

💡 Usage Notes
- Place your trained model in Assets/Model/mobilenetv2.h5
- Add your test images in Assets/Test/ for testing from the dataset tab
- Upload any image via the "Upload Image" tab to classify it
