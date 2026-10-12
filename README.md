# BABY-CRY-CLASSIFICATION
Baby Cry Classification is a machine learning project developed to identify different categories of baby cries using audio recordings. The project uses a Random Forest Classifier to learn patterns from audio signals and predict the category associated with a given recording.

The system extracts important audio features using Librosa and NumPy, including frequency components, spectral centroid, spectral bandwidth, and zero-crossing rate. These features help represent the characteristics of the audio and are used as input to the machine learning model.

The project uses Python and Scikit-learn for data preprocessing, model training, testing, and hyperparameter tuning. The dataset is divided into training and testing sets to evaluate the model's performance. RandomizedSearchCV is used to explore different model configurations and identify suitable hyperparameters. The trained model is saved as a pickle file for future use.

The project also includes a Gradio interface that provides a simple way for users to interact with the system. This project demonstrates the practical application of machine learning and audio feature extraction for classifying baby cry patterns.


