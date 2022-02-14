# Optical Character Recognition

Real-life handwritten characters are different from printed characters because every person has a unique writing style and recognizing them is a challenging problem for computers. Optical Character Recognition (OCR) solves this type of problem and in this project SVM and KNN methods were applied to the recognition of [handwritten letters](http://ai.stanford.edu/~btaskar/ocr/). Before applying these methods, the dataset was processed with different techniques such as Scaling and Principal Component Analysis. They reduced features while retaining features that have more impact on the shape of letters. Dataset trained and tested with K-Fold Cross Validation method which is an effective way to work with the dataset and helps to reduce bias for training. Test results show that for specified configurations in the project SVM has better accuracy compared to the KNN counterpart.