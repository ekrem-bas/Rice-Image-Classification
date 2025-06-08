# Rice Image Classification: A CNN vs. Random Forest Comparison

This project aims to classify images of rice grains into six different types: Arborio, Basmati, Ipsala, Jasmine, Karacadag, and Osmancik. For this classification task, two different machine learning models are developed, trained, and compared: a **Convolutional Neural Network (CNN)** and a **Random Forest** classifier.

## 📝 Project Overview

Using image processing and machine learning techniques, a system was developed to predict the type of a rice grain from its image. The project evaluates the models based on metrics such as **Accuracy**, **F1-Score**, **ROC-AUC Score**, and the **confusion matrix** to provide an objective comparison of their performance.

---

## 📂 Project File Structure

```
.
├── main.ipynb                  # Main Jupyter Notebook containing all project steps
├── requirements.txt            # Required Python libraries
├── data_direction.py           # Configuration file to specify the dataset path
├── cnn_model.pth               # Saved state of the trained CNN model
└── random_forest_model.joblib  # Saved state of the trained Random Forest model
```

---

## 🛠️ Technologies & Libraries Used

- **Python 3.x**
- **PyTorch & Torchvision:** For designing, training, and evaluating the CNN model.
- **Scikit-learn:** For building the Random Forest model and calculating performance metrics (Confusion Matrix, Classification Report, F1-Score, etc.).
- **Numpy & Pandas:** For data manipulation, processing, and presenting results in a structured format.
- **Matplotlib & Seaborn:** For visualizing training/test results and data analysis (e.g., loss/accuracy graphs, confusion matrices).
- **Joblib:** For saving and loading the trained Random Forest model.

---

## 🚀 Setup and Execution

Follow these steps to run the project on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/ekrem-bas/Image-Classification.git
cd Image-Classification
```

### 2. Install Required Libraries
All project dependencies are listed in the `requirements.txt` file. Install them using pip.
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
This project uses the **[Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)** from Kaggle. Please download the dataset and extract it to a folder.

### 4. Create the Data Direction File
The project reads the dataset path from a file named `data_direction.py`. Create this file in the project's root directory and add the following line. Replace `"/path/to/your/dataset"` with the actual path to your dataset folder.

**`data_direction.py` content:**
```python
# Update the path below to your dataset's location
DATA_DIR = "/path/to/your/dataset/Rice_Image_Dataset"
```
The dataset folder structure should be as follows:
```
Rice_Image_Dataset/
├── train/
│   ├── Arborio/
│   ├── ... (other classes)
└── test/
    ├── Arborio/
    └── ... (other classes)
```

### 5. Run the Notebook
Once all setup steps are complete, open the `main.ipynb` file in a Jupyter Notebook or Jupyter Lab environment and run all cells sequentially. The code will automatically detect and use hardware accelerators like **CUDA** or **MPS (Apple Silicon)** if available.

---

## 🤖 Models and Methodology

### 1. Convolutional Neural Network (CNN)

This deep learning-based model is designed to learn hierarchical features from image data.

- **Architecture Details**:
  - **Input:** 3-channel (RGB) images, resized to 128x128 pixels and normalized.
  - **Convolutional Layers:**
    1.  `Conv2d (3x3, 32 filters)` -> `ReLU` -> `MaxPool2d (2x2)`
    2.  `Conv2d (3x3, 64 filters)` -> `ReLU` -> `MaxPool2d (2x2)`
    3.  `Conv2d (3x3, 128 filters)` -> `ReLU` -> `MaxPool2d (2x2)`
  - **Fully Connected Layers:**
    1.  `Linear (32768 -> 128)` -> `ReLU`
    2.  `Linear (128 -> 64)` -> `ReLU`
    3.  `Dropout (p=0.3)`
    4.  `Linear (64 -> 6)` (Output layer for 6 classes)
- **Training Parameters**:
  - **Optimizer:** Adam (with L2 Regularization)
  - **Loss Function:** Cross-Entropy Loss
  - **Epochs:** 10

### 2. Random Forest

As a classical machine learning approach, this model performs classification after feature extraction.

- **Feature Extraction**: Feature vectors were created for each image using pixel intensity histograms with 256 bins. This method simplifies the image's structural information into a more basic format.
- **Model Parameters**:
  - `n_estimators`: 100
  - `max_depth`: 10
  - `min_samples_split`: 25
- **Data Splitting**: The dataset was split into 80% training and 20% testing sets, preserving the class distribution (stratified split).

---

## 📊 Results and Evaluation

The performance of both models on the test data is detailed below.

### Performance Metrics Comparison

| Model | ROC-AUC | F1 Score | Accuracy | Training Time (sec) |
| :--- | :---: | :---: | :---: | :---: |
| **CNN** | **0.9999** | **0.976** | **0.976** | 16.11 |
| **Random Forest** | 0.8995 | 0.670 | 0.675 | **0.06** |

*The values in the table are based on the outputs from the provided `main.ipynb` file*.

### Analysis
- **CNN Model:** The CNN achieved an outstanding accuracy of 97.6%. An ROC-AUC score of 0.9999 indicates a near-perfect ability to distinguish between classes. The classification report confirms that the model identifies all rice types with high precision and recall.

- **Random Forest Model:** Although significantly faster to train, the Random Forest model's accuracy of 67.5% lags far behind the CNN. The confusion matrix shows that it struggles to differentiate between more visually similar classes like *Arborio* and *Basmati*. This suggests that simple features like histograms are insufficient for capturing the complex patterns in the images.

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
__

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

__

## ✨ Conclusion

This study demonstrates that for complex image processing tasks like rice type classification, **Convolutional Neural Networks (CNNs)** deliver **far superior performance** compared to classical machine learning methods. Despite the longer training time, the high accuracy of the CNN proves it is the more suitable model for this task.

## 💾 Saved Models
After the training process is complete, the weights of both models are saved to the root directory for future use:
- **CNN Model:** `cnn_model.pth`
- **Random Forest Model:** `random_forest_model.joblib`
