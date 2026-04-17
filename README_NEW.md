# Sign Language Recognition - Minor Project

This project implements multiple machine learning and deep learning approaches for recognizing American Sign Language (ASL) alphabet signs from images. The project contains four different models using various techniques for comparison and evaluation.

## Project Overview

The objective of this project is to classify ASL alphabet signs using different machine learning algorithms. All models are trained and evaluated on the ASL Alphabet Dataset from Kaggle, which contains images of hand gestures representing the 26 letters of the English alphabet plus additional symbols.

## Dataset

- **Source**: Kaggle - ASL American Sign Language Alphabet Dataset (debashishsau/aslamerican-sign-language-aplhabet-dataset)
- **Training Directory**: asl_alphabet_train
- **Test Directory**: asl_alphabet_test
- **Classes**: 29 classes (26 letters A-Z plus 3 additional symbols)
- **Image Format**: JPG/PNG
- **Preprocessing**: Images are resized to specific dimensions depending on the model

## Project Structure

```
Mobile_net_v2.ipynb
random_forest_sign.ipynb
Sign_Language_Recognition_cnn.ipynb
Sign_Language_recognition_svm.ipynb
README.md
```

## Models Implemented

### 1. Mobile_net_v2.ipynb - MobileNetV2 Transfer Learning

**Approach**: Deep Learning with Transfer Learning

**Description**:
- Uses a pre-trained MobileNetV2 model from ImageNet
- Implements transfer learning to classify ASL signs
- Freezes the base model initially, then fine-tunes the top layers
- Uses ImageDataGenerator for data augmentation
- Input image size: 224 x 224 pixels
- Batch size: 32

**Key Features**:
- Global Average Pooling layer for dimensionality reduction
- Dense layer with 256 units and ReLU activation
- Dropout (0.5) for regularization
- Softmax output layer for multi-class classification
- Adam optimizer with learning rate 0.0001
- Categorical crossentropy loss function

**Advantages**:
- Leverages pre-trained weights from ImageNet
- Fast convergence due to transfer learning
- Memory-efficient architecture (MobileNetV2)
- Good accuracy with minimal training time

---

### 2. random_forest_sign.ipynb - Random Forest Classifier

**Approach**: Machine Learning with Handcrafted Features

**Description**:
- Extracts HOG (Histogram of Oriented Gradients) features from images
- Trains a Random Forest classifier on the extracted features
- Uses scikit-learn's RandomForestClassifier
- Image size: 64 x 64 pixels
- HOG parameters: 9 orientations, 8x8 pixels per cell, 2x2 cells per block

**Key Process**:
1. Image loading and grayscale conversion
2. Image resizing to 64 x 64
3. HOG feature extraction using scikit-image
4. Data limitation (700 samples per class for computational efficiency)
5. Train-test split (80-20)
6. Model training and evaluation

**Evaluation Metrics**:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report

**Advantages**:
- Computationally efficient
- Good for quick prototyping and testing
- Interpretable feature extraction
- No GPU required

---

### 3. Sign_Language_Recognition_cnn.ipynb - Custom CNN

**Approach**: Deep Learning with Custom Architecture

**Description**:
- Builds a custom Convolutional Neural Network from scratch
- Uses TensorFlow/Keras Sequential API
- Designed specifically for ASL recognition tasks
- Implements regularization techniques

**Architecture Components**:
- Convolutional layers with varying filters
- MaxPooling layers for spatial downsampling
- BatchNormalization for stable training
- Dense layers for classification
- Dropout for regularization
- EarlyStopping callback for preventing overfitting
- ReduceLROnPlateau callback for learning rate adjustment

**Key Features**:
- Data augmentation via ImageDataGenerator
- Visualization of dataset distribution
- Comprehensive training history and loss tracking
- Confusion matrix and detailed accuracy metrics
- Support for real-time prediction

**Advantages**:
- Full control over architecture design
- Customizable for specific problem requirements
- Better interpretability compared to transfer learning
- Can achieve high accuracy with proper tuning

---

### 4. Sign_Language_recognition_svm.ipynb - Support Vector Machine

**Approach**: Machine Learning with Dimensionality Reduction

**Description**:
- Extracts HOG features from images
- Applies PCA (Principal Component Analysis) for dimensionality reduction
- Trains a LinearSVC (Support Vector Machine) classifier
- Image size: 64 x 64 pixels
- PCA components: 120

**Key Process**:
1. HOG feature extraction (same as Random Forest approach)
2. Data limitation (700 samples per class)
3. PCA for dimensionality reduction (120 components selected)
4. Train-test split (80-20)
5. LinearSVC model training with C=1.0, max_iter=6000
6. Comprehensive evaluation and metrics

**Evaluation Metrics**:
- Accuracy
- Precision, Recall, F1-Score (macro average)
- Confusion Matrix
- Classification Reports

**Advantages**:
- Effective for binary and multi-class classification
- PCA helps in handling high-dimensional feature data
- Better performance than Random Forest on some tasks
- Scales well to high-dimensional data

---

## System Requirements

**Python Libraries**:
- TensorFlow / Keras (for deep learning models)
- scikit-learn (for SVM and Random Forest)
- OpenCV (cv2) - for image processing
- NumPy - for numerical computations
- Pandas - for data manipulation
- Matplotlib - for visualization
- Seaborn - for advanced visualization
- scikit-image - for HOG feature extraction
- JobLib - for model persistence
- KaggleHub - for dataset download
- tqdm - for progress bars

**Hardware**:
- GPU support recommended for MobileNetV2 and CNN models
- CPU is sufficient for Random Forest and SVM models
- Minimum 4GB RAM required

## Running the Project

### Prerequisites

1. Install required packages:
   ```bash
   pip install tensorflow opencv-python scikit-learn pandas matplotlib seaborn scikit-image joblib kagglehub tqdm numpy
   ```

2. Install Jupyter Notebook or JupyterLab:
   ```bash
   pip install jupyter
   ```

### Execution Steps

1. Open any of the four Jupyter notebooks
2. Run the first cell to import the Kaggle dataset using KaggleHub
3. Execute cells sequentially to:
   - Load and preprocess the dataset
   - Extract features (if applicable)
   - Train the model
   - Evaluate performance
   - Generate visualizations

4. Save trained models for future predictions

## Model Training Parameters

### MobileNetV2
- Image Size: 224 x 224
- Batch Size: 32
- Initial Epochs: 10
- Learning Rate: 0.0001
- Validation Split: 0.2 (20%)

### Random Forest
- Image Size: 64 x 64
- Samples per Class: 700
- Test Size: 20%
- HOG Orientations: 9

### Custom CNN
- Image Size: Variable
- Batch Size: 32
- Epochs: Variable (with EarlyStopping)
- Learning Rate: Adaptive (ReduceLROnPlateau)

### SVM
- Image Size: 64 x 64
- Samples per Class: 700
- PCA Components: 120
- Test Size: 20%
- SVM C Parameter: 1.0
- Max Iterations: 6000

## Performance Comparison

The project allows for direct comparison of different approaches:

- **Deep Learning Models** (MobileNetV2, CNN) - Generally provide higher accuracy
- **Machine Learning Models** (Random Forest, SVM) - Faster training, lower computational cost

Each model can be evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Feature Extraction Methods

### HOG (Histogram of Oriented Gradients)
Used in Random Forest and SVM models for capturing edge and shape information

Parameters:
- Window Size: 64 x 64
- Block Size: 16 x 16
- Block Stride: 8 x 8
- Cell Size: 8 x 8
- Number of Bins: 9

### Deep Features
Used in MobileNetV2 and CNN models for automatic feature learning

## Data Preprocessing

- **Image Resizing**: Each model resizes images to its required dimensions
- **Normalization**: Color normalization applied (model-specific)
- **Augmentation**: Used in CNN and MobileNetV2 for improved generalization
- **Grayscale Conversion**: Applied in HOG-based models for efficiency

## Output and Visualization

All models generate:
- Accuracy plots
- Confusion matrices
- Classification reports
- Precision, Recall, F1-Score metrics
- Training history (for deep learning models)
- Feature distribution plots

## Model Saving and Loading

Trained models are saved using:
- pickle / joblib format (for sklearn models)
- SavedModel or HDF5 format (for TensorFlow models)

This allows for reuse without retraining.

## Future Improvements

1. Implement ensemble methods combining multiple models
2. Deploy models to web or mobile applications
3. Add real-time hand gesture recognition from camera feed
4. Extend to recognize hand positions and movements (not just static signs)
5. Implement attention mechanisms for improved accuracy
6. Perform hyperparameter tuning for optimal performance
7. Add support for video-based sign language recognition

## Project Objectives

- Compare multiple machine learning and deep learning approaches
- Evaluate performance metrics across different models
- Understand the trade-offs between accuracy and computational efficiency
- Develop practical sign language recognition expertise
- Create reusable models for ASL recognition tasks

## Notes

- All notebooks import data from Kaggle using KaggleHub
- Models can be adapted for other image classification tasks
- The project demonstrates both classical ML and modern deep learning techniques
- Careful attention to data preprocessing ensures consistent results across models

## License

This project uses the ASL Alphabet Dataset from Kaggle (debashishsau/aslamerican-sign-language-aplhabet-dataset).

## Authors

Minor project on Sign Language Recognition using Multiple ML/DL Approaches
