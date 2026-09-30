# PneumoScan-AI
# 🫁 LungLens – Pneumonia Detection from Chest X-Rays

LungLens is a deep learning web app that screens chest X-ray images for signs of pneumonia. Upload an X-ray, and the model predicts **Normal** or **Pneumonia** with a confidence score, plus basic guidance if pneumonia is detected.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is not a medical device and must not be used for diagnosis. Always consult a certified medical professional.

---

## ✨ Features

- Upload chest X-rays (JPG, JPEG, PNG)
- Binary classification: Normal vs. Pneumonia
- Confidence score with a progress bar
- Medical guidance panel (symptoms, recommended action, precautions) when pneumonia is detected
- Clean Streamlit interface with a two-panel layout

## 🧠 How It Works

1. The uploaded image is converted to RGB, resized to 224×224, and normalized with ImageNet statistics.
2. A **ResNet18** CNN, with its final layer replaced by a 2-class output, processes the image.
3. Softmax converts the output into class probabilities.
4. The app shows the predicted class and its confidence.

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Deep learning | PyTorch, torchvision (ResNet18) |
| Web app | Streamlit |
| Image handling | Pillow |
| Language | Python 3.9+ |

## 📁 Project Structure

```
lunglens/
├── app.py                 # Streamlit application
├── pneumonia_model.pth    # Trained model weights
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

**1. Clone the repo**

```bash
git clone https://github.com/<your-username>/lunglens.git
cd lunglens
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add the trained weights**

Place `pneumonia_model.pth` in the project root. If the file is missing, the app still runs, but the model will use untrained weights and the predictions will not be meaningful.

**4. Run the app**

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## 📦 requirements.txt

```
streamlit
torch
torchvision
Pillow
```

## 📊 Dataset

Trained on a public Chest X-Ray Images (Pneumonia) dataset from Kaggle.

## 📈 Model Performance

_Add your results here (accuracy, precision, recall, F1, confusion matrix) after evaluating on the test set._

## 🔮 Future Improvements

- Grad-CAM heatmaps to show which lung regions influenced the prediction
- Handle class imbalance and add data augmentation
- Add a check to reject non-X-ray images
- Compare against other architectures (ResNet50, EfficientNet, DenseNet)
- Deploy on Streamlit Community Cloud or Hugging Face Spaces

## 👩‍💻 Author

**Akanksha Reddy Gorla**
M.S. in Artificial Intelligence, University of North Texas
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)

## 📄 License

Released under the MIT License.
