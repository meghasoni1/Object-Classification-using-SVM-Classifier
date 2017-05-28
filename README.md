# Object-Classification-using-SVM-Classifier
Python implementation of SVM Classifier using Bag of Visual Words.

## **TRAINING THE CLASSIFIER**
SIFT features are extracted from the directory of images. Training of the classifier is done using the LinearSVC kernel of SVM from the scikit-learn library.

> python getFeatures.py -t dataset/train/

##**TESTING THE CLASSIFIER**
Classification is carried out on the directory of testing images using the classifier trained earlier.

> python classify.py -t dataset/test --visualize

