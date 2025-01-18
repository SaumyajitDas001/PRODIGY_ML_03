# PRODIGY_ML_03
A Cat and Dog Classifier system Using SVM Learning.
This project involves building a **Cat and Dog Classifier** using a **Support Vector Machine (SVM)** model. The classifier aims to differentiate between images of cats and dogs based on pixel features.

The dataset comprises labeled images of cats and dogs stored in separate directories. Each image is resized to a uniform dimension of 64x64 pixels for consistency, normalized to scale pixel values to a range of [0, 1], and flattened into a one-dimensional array to serve as input for the SVM.

The data is split into training and testing sets, with 80% of the data used for training and 20% reserved for evaluation. A **linear kernel SVM** is employed, optimized to maximize the margin between the two classes. To enhance performance, the features are standardized using a **StandardScaler**, a crucial step for SVM-based models.

After training, the model’s performance is evaluated using accuracy and a classification report, highlighting its effectiveness in distinguishing between cats and dogs. While the current implementation uses raw pixel features, further enhancements like employing convolutional neural network (CNN) embeddings as inputs can significantly improve classification accuracy.

This project demonstrates the power of SVM in image classification tasks and emphasizes the importance of preprocessing and feature scaling for effective performance.
