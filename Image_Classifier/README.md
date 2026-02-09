# CIFAR-10 Image Classification with Keras and TensorFlow

This notebook demonstrates a comprehensive approach to image classification on the CIFAR-10 dataset using various machine learning models, culminating in a highly optimized Convolutional Neural Network (CNN).

## Table of Contents
1.  [Introduction](#introduction)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Traditional Machine Learning Classifiers](#traditional-machine-learning-classifiers)
    *   [Simple Neural Networks](#simple-neural-networks)
    *   [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
        *   [Progressive CNN Architectures](#progressive-cnn-architectures)
        *   [Hyperparameter Tuning](#hyperparameter-tuning)
4.  [Results](#results)
5.  [Conclusion](#conclusion)
6.  [Installation and Usage](#installation-and-usage)

## Introduction
This project aims to classify images from the CIFAR-10 dataset into one of 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. We explore different classification techniques, starting from traditional machine learning models and progressing to more sophisticated deep learning architectures, specifically Convolutional Neural Networks, to achieve high accuracy.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Methodology

### Data Preprocessing
-   **Loading Data**: The CIFAR-10 dataset is loaded directly from Keras.
-   **Reshaping Labels**: The `y_train` and `y_test` labels are flattened from 2D arrays to 1D arrays.
-   **Normalization**: Pixel values of the images are scaled from the range [0, 255] to [0, 1] by converting to `float32` and dividing by 255.
-   **Data Inspection**: The shapes of the training and testing datasets are checked, and the distribution of classes is verified to ensure balance.

### Traditional Machine Learning Classifiers
Initial classification attempts were made using standard scikit-learn classifiers:
-   `SVC` (Support Vector Classifier)
-   `DecisionTreeClassifier`
-   `ExtraTreeClassifier`
-   `RandomForestClassifier`

These models were applied to reshaped image data (flattened 32x32x3 to 3072 features). While `SVC` yielded the best performance among these (54% validation accuracy), traditional methods proved suboptimal for image classification due to their inability to effectively capture spatial hierarchies.

### Simple Neural Networks
Several simple dense neural network architectures were explored with varying numbers of layers and neurons. Hyperparameters like dropout rate and batch size were also varied. The best simple neural network achieved approximately 59% validation accuracy, demonstrating an improvement over traditional methods but still insufficient for robust image classification.

### Convolutional Neural Networks (CNNs)
Recognizing the limitations of simpler models, the focus shifted to CNNs, which are specifically designed for image data.

#### Progressive CNN Architectures
The notebook progressively builds more complex CNN models:
1.  **Single Conv Layer**: A basic CNN with one `Conv2D` layer, `BatchNormalization`, `Dropout`, `Flatten`, and `Dense` layers significantly improved accuracy to 64%.
2.  **Two Conv Layers + MaxPooling**: Adding another `Conv2D` layer and a `MaxPooling2D` layer further boosted performance to 75% validation accuracy.
3.  **Three Conv Layers + Two MaxPooling**: Increasing to three `Conv2D` layers and two `MaxPooling2D` layers, along with additional `Dense` layers, pushed validation accuracy to 82%.
4.  **Six Conv Layers + Three MaxPooling (Optimized)**: The most complex architecture involved three blocks, each with two `Conv2D` layers, `BatchNormalization`, `MaxPooling2D`, and `Dropout`. This configuration achieved an impressive 89% validation accuracy.

#### Hyperparameter Tuning
Extensive hyperparameter tuning was performed on the best CNN architecture:
-   **Kernel Size**: Tested `kernel_size` from 3 to 7. The optimal was found to be 3.
-   **Batch Size**: Explored `batch_size` values like 128, 256, 512, 1024, 4096. The best performance was observed with `batch_size = 256`.
-   **Dropout Value**: Different dropout rates (e.g., 0.1, 0.2, 0.3, 0.4, 0.5) were evaluated. A dropout of 0.4 consistently provided the best results.
-   **Learning Rate**: Various learning rates (e.g., 0.03, 0.02, 0.01, 0.005, 0.0025) were tested. An Adam optimizer with a `learning_rate` of 0.01 yielded the highest accuracy.
-   **Filter Configuration**: Experiments with fewer filters (e.g., 32, 64, 128 instead of 64, 128, 256) were conducted to assess model size vs. accuracy tradeoffs. While lighter models were created, they generally had slightly lower accuracy.
-   **Increased Conv Layers/Filters**: Attempts to increase the number of convolutional layers (e.g., 9 `Conv2D` layers in total, 3 per block) and filters (up to 512) were made, but did not significantly improve validation accuracy beyond the optimized 6-layer CNN while substantially increasing model complexity.

## Results
After thorough experimentation and hyperparameter tuning, the final optimized CNN model achieved a validation accuracy of **90.3%** on the CIFAR-10 dataset. This model configuration provided the highest accuracy while maintaining a reasonable level of complexity.

-   **Model Visualization**: The architecture of the best model is visualized using `visualkeras`.
-   **Confusion Matrix**: A confusion matrix is generated to analyze the model's performance on individual classes. The model performed best on classifying `automobiles`, `trucks`, `ships`, and `frogs`, while struggling most with `cats`, `dogs`, and `birds` (often confusing `dogs` and `cats`).

## Conclusion
Convolutional Neural Networks proved to be highly effective for image classification on the CIFAR-10 dataset, significantly outperforming traditional machine learning and simple neural networks. Hyperparameter tuning played a crucial role in optimizing the model's performance. Further improvements could potentially be achieved through data augmentation or more advanced CNN architectures.

## Installation and Usage

To run this notebook, you will need the following Python libraries:
-   `tensorflow`
-   `numpy`
-   `pandas`
-   `matplotlib`
-   `visualkeras`
-   `scikit-learn`

You can install them using pip:
```bash
pip install tensorflow numpy pandas matplotlib visualkeras scikit-learn
```

Then, open the `.ipynb` file in a Jupyter environment (like Google Colab or Jupyter Notebook) and run all cells.