# Rice Image Classification: A CNN vs. Random Forest Comparison

## 📝 Project Overview

This project classifies images of rice grains into six types: Arborio, Basmati, Ipsala, Jasmine, Karacadag, and Osmancik. It compares the performance of a Convolutional Neural Network (CNN) and a Random Forest classifier using image processing and machine learning techniques. The models are evaluated using metrics such as Accuracy, F1-Score, ROC-AUC Score, and confusion matrix.

## 📂 Project Structure

```
.
├── main.ipynb                  # Main Jupyter Notebook containing all project steps
├── requirements.txt            # Required Python libraries
├── data_direction.py           # Configuration file to specify the dataset path
├── cnn_model.pth               # Saved state of the trained CNN model
└── random_forest_model.joblib  # Saved state of the trained Random Forest model
```

## 🛠️ Technologies & Libraries Used

- Python 3.x
- PyTorch & Torchvision: CNN model design, training, and evaluation
- Scikit-learn: Random Forest model and performance metrics
- Numpy & Pandas: Data manipulation and processing
- Matplotlib & Seaborn: Data and result visualization
- Joblib: Model serialization (Random Forest)

## 🚀 Setup and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/ekrem-bas/Rice-Image-Classification.git
cd Rice-Image-Classification
```

### 2. Install Required Libraries

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download the [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) from Kaggle and extract it to a folder on your machine.

The dataset folder structure should be:

```
Rice_Image_Dataset/
├── train/
│   ├── Arborio/
│   ├── ... (other classes)
└── test/
    ├── Arborio/
    └── ... (other classes)
```

### 4. Create the Data Direction File

Create a file named `data_direction.py` in the project root with the following content (update the path):

```python
DATA_DIR = "/path/to/your/dataset/Rice_Image_Dataset"
```

### 5. Run the Notebook

Open `main.ipynb` in Jupyter Notebook or Jupyter Lab and run all cells sequentially. The code will use hardware accelerators (CUDA/MPS) if available.

## 🤖 Models and Methodology

### 1. Convolutional Neural Network (CNN)

- **Input:** 3-channel RGB images, resized to 128x128 pixels, normalized
- **Architecture:**
  - Conv2d (3x3, 32 filters) → ReLU → MaxPool2d (2x2)
  - Conv2d (3x3, 64 filters) → ReLU → MaxPool2d (2x2)
  - Conv2d (3x3, 128 filters) → ReLU → MaxPool2d (2x2)
  - Linear (32768 → 128) → ReLU
  - Linear (128 → 64) → ReLU
  - Dropout (p=0.3)
  - Linear (64 → 6) (output layer)
- **Training:** Adam optimizer (L2 regularization), Cross-Entropy Loss, 10 epochs

### 2. Random Forest

- **Feature Extraction:** 256-bin pixel intensity histograms per image
- **Parameters:**
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 25
- **Data Split:** 80% train, 20% test (stratified)

## 📊 Results and Evaluation

### Performance Metrics Comparison

| Model         | ROC-AUC | Accuracy | F1-Score | Training Time (min) |
| ------------- | ------- | -------- | -------- | ------------------- |
| CNN           | 0.9999  | 0.976    | 0.976    | 16.11               |
| Random Forest | 0.8995  | 0.670    | 0.675    | 0.06                |

### Detailed Classification Report (CNN)

```
              precision    recall  f1-score   support

     Arborio       1.00      0.90      0.95        21
     Basmati       1.00      0.95      0.98        21
      Ipsala       1.00      1.00      1.00        21
     Jasmine       0.95      1.00      0.98        21
   Karacadag       0.91      1.00      0.95        21
    Osmancık       1.00      1.00      1.00        21

    accuracy                           0.98       126
   macro avg       0.98      0.98      0.98       126
weighted avg       0.98      0.98      0.98       126
```

### Detailed Classification Report (Random Forest)

```
              precision    recall  f1-score   support

     Arborio       0.38      0.30      0.33        20
     Basmati       0.26      0.25      0.26        20
      Ipsala       1.00      1.00      1.00        20
     Jasmine       0.86      0.90      0.88        20
   Karacadag       0.52      0.65      0.58        20
    Osmancık       1.00      0.95      0.97        20

    accuracy                           0.68       120
   macro avg       0.67      0.67      0.67       120
weighted avg       0.67      0.68      0.67       120
```

### Analysis

- **CNN Model:** Achieved 97.6% accuracy and near-perfect ROC-AUC (0.9999). High precision and recall for all classes.
- **Random Forest Model:** Faster to train but only 67.5% accuracy. Struggles with visually similar classes, indicating histogram features are insufficient for complex image patterns.

## ✨ Conclusion

Convolutional Neural Networks (CNNs) significantly outperform classical machine learning methods like Random Forests for complex image classification tasks. While CNNs require more training time, their superior accuracy makes them the preferred choice for rice type classification.

## 💾 Saved Models

- `cnn_model.pth`: Trained CNN model weights
- `random_forest_model.joblib`: Trained Random Forest model

## 📚 References

- [Rice Image Dataset on Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

For questions or contributions, please open an issue or pull request on the [GitHub repository](https://github.com/ekrem-bas/Rice-Image-Classification).
