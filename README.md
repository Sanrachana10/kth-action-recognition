# 🎥 KTH Human Action Recognition using CNN-LSTM

A deep learning-based Human Action Recognition system that classifies video clips into one of six actions using a hybrid CNN-LSTM architecture. The model captures both spatial features from individual frames and temporal motion patterns across video sequences.

## 🚀 Features

- Human action recognition from videos
- CNN-based spatial feature extraction
- LSTM-based temporal sequence modeling
- Streamlit web application for inference
- Confidence score visualization
- Support for video upload and real-time prediction

## 📊 Dataset

This project uses the **KTH Human Action Recognition Dataset**, a standard benchmark dataset for action recognition.

### Action Classes

- Boxing
- Handclapping
- Handwaving
- Jogging
- Running
- Walking

### Dataset Statistics

- 600 video sequences
- 25 subjects
- 4 recording scenarios
- Grayscale videos
- 25 FPS

## 🏗️ Model Architecture

The model follows a CNN-LSTM pipeline:

```text
Input Video
     ↓
Frame Extraction (15 Frames)
     ↓
Preprocessing
(Grayscale + Resize 64×64)
     ↓
TimeDistributed CNN
     ↓
Feature Sequence
     ↓
LSTM
     ↓
Dense Layer + Softmax
     ↓
Predicted Action
```

### Design Choices

- 15 evenly sampled frames per video
- Frame size: 64 × 64
- Grayscale processing
- CNN for spatial feature learning
- LSTM for temporal dependency modeling
- Softmax classifier for multi-class prediction

## 📈 Results

| Metric | Score |
|----------|--------|
| Accuracy | 72% |
| Macro Precision | 0.75 |
| Macro Recall | 0.72 |
| Macro F1 Score | 0.73 |

### Best Performing Class
- Boxing (F1 Score: 0.95)

### Main Challenge
- Jogging and Running frequently get confused due to highly similar motion patterns.

## 🛠️ Tech Stack

### Deep Learning
- TensorFlow
- Keras

### Computer Vision
- OpenCV

### Data Processing
- NumPy
- Pandas
- Scikit-learn

### Deployment
- Streamlit
- Hugging Face Hub

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/kth-action-recognition.git
cd kth-action-recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 💻 Usage

1. Launch the Streamlit application.
2. Upload a video file (.mp4, .avi, .mov).
3. The system extracts 15 frames automatically.
4. The CNN-LSTM model predicts the action.
5. View confidence scores for all six classes.


## 🔮 Future Improvements

- Bidirectional LSTM
- 3D CNN architectures
- Video Transformers
- Real-time webcam inference
- Evaluation on UCF101 and HMDB51
- Higher-resolution frame inputs

## 👨‍💻 Author

**Sanrachana Singh**

Information Technology Undergraduate, SGSITS Indore

Interests:
- Artificial Intelligence & Machine Learning
- Computer Vision
- NLP & RAG Systems
- Backend Development

---

⭐ If you found this project interesting, consider giving it a star!
