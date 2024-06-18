# Obesity Level Risk Prediction

## Overview

This project aims to predict the obesity levels of individuals based on various features such as age, height, weight, and lifestyle factors. Multiple machine learning models are utilized, including Logistic Regression, Support Vector Machine (SVM), and ensemble methods like Bagging with SVM. The project involves thorough data preprocessing to handle categorical and numerical features appropriately before training the models.

## Table of Contents
1. [Data Import and Libraries](#data-import-and-libraries)
2. [Data Exploration](#data-exploration)
3. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
   - [Encoding Categorical Columns](#encoding-categorical-columns)
   - [Standardizing Numerical Columns](#standardizing-numerical-columns)
4. [Model Training and Evaluation](#model-training-and-evaluation)
   - [Logistic Regression](#logistic-regression)
   - [Support Vector Machine (SVM)](#support-vector-machine-svm)
   - [Bagging Classifier with SVM](#bagging-classifier-with-svm)
5. [List of Models](#list-of-models)
6. [Conclusion](#conclusion)

## Data Import and Libraries

In the initial step, essential libraries for data manipulation, visualization, and machine learning are imported. The dataset is loaded into a DataFrame for analysis.

## Data Exploration

A thorough exploration of the dataset is conducted to understand the distribution of features, check for missing values, and identify any anomalies. Visualizations such as histograms, box plots, and scatter plots are created to gain insights into the data.

## Data Cleaning and Preprocessing

### Encoding Categorical Columns

Categorical columns are transformed into numerical values using label encoding. This step converts categories into integers, enabling machine learning algorithms to process these features. A common error to watch for is ensuring that the transformation is applied correctly to each column individually.

### Standardizing Numerical Columns

Numerical columns are standardized to have a mean of zero and a standard deviation of one. This step is crucial for algorithms like SVM, which are sensitive to the scale of input data. The standardization process ensures that all numerical features contribute equally to the model.

## Model Training and Evaluation

### Logistic Regression

A logistic regression model is trained to classify obesity levels. The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. The results provide a baseline for comparing more complex models.

### Support Vector Machine (SVM)

An SVM model is trained with a radial basis function (RBF) kernel. This model is known for its effectiveness in high-dimensional spaces. The SVM's performance is assessed using the same metrics as the logistic regression model.

### Bagging Classifier with SVM

An ensemble method, Bagging Classifier with SVM as the base estimator, is implemented. Bagging involves training multiple instances of the base model on different subsets of the data and averaging their predictions. This technique helps in reducing overfitting and improving model robustness. Common issues like the compatibility of the base estimator's parameters are addressed.

## List of Models

1. **Logistic Regression**
   - A linear model used for binary classification, serving as a baseline for performance comparison.
   
2. **Support Vector Machine (SVM)**
   - An SVM model with an RBF kernel is used, which is effective in high-dimensional spaces and for non-linear classification tasks.

3. **Bagging Classifier with SVM**
   - An ensemble learning method that uses Bagging with SVM as the base estimator to reduce overfitting and improve the robustness of predictions.

## Conclusion

The project demonstrates the application of various machine learning techniques to predict obesity levels based on multiple features. Through data exploration, preprocessing, and model training, we achieve a comprehensive understanding of the dataset and build effective predictive models. Future work could involve exploring other ensemble methods or deep learning approaches to further enhance prediction accuracy.
