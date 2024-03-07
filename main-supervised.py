"""
Supervised Model Training for Credit Card Fraud Detection Using Pseudo Labels

This module presents an innovative approach to credit card fraud detection by leveraging pseudo labels generated from an unsupervised Variational Autoencoder (VAE) model. After training the VAE on unlabeled data to capture the underlying data distribution and anomalies, this module employs the VAE's reconstructions to generate pseudo labels for the training data. These pseudo labels serve as an informed guess of the data's fraudulence, allowing for the training of a supervised XGBoost classifier with a more nuanced understanding of fraudulent patterns than traditional methods.

The process involves:
- Preprocessing the dataset to standardize features, facilitating more effective learning.
- Evaluating the VAE model's reconstruction error to generate pseudo labels, with higher errors indicating potential fraud.
- Utilizing an XGBoost model, trained on the data labeled by the VAE, to classify transactions as fraudulent or legitimate.
- Employing various performance metrics, including precision, recall, F1 score, PR AUC, and ROC AUC, to assess the effectiveness of the fraud detection system.

By integrating the generative capabilities of VAEs with the predictive power of XGBoost, this module aims to enhance the detection of credit card fraud, showcasing a novel blend of unsupervised and supervised machine learning techniques for security and anomaly detection tasks.
"""


# Import libraries
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.metrics import make_scorer
from torch.utils.data import DataLoader
from source import CreditCardDataset, IntroVAE, StandardScalerTransform

# Set the device to GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Evaluate the Performance of the IntroVAE Model
# This function calculates and plots key evaluation metrics, including precision-recall and ROC curves,
# to assess the model's ability to distinguish between fraudulent and legitimate transactions accurately.

# Function to evaluate the model
def evaluate_model(model, dataloader, device):
    # Set the model to evaluation mode
    model.eval()  
    
    labels, scores = [], []
    # Loop over the data
    with torch.no_grad():
        for inputs, targets, _ in dataloader:  
            # Move inputs and targets to the correct device
            inputs, targets = inputs.to(device), targets.to(device)  
            # Get the model's reconstruction of the inputs and calculate the error
            reconstruction, _, _ = model(inputs)
            error = torch.mean((inputs - reconstruction) ** 2, dim=1)
            # Extend the scores and labels lists
            scores.extend(error.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    # Convert scores and labels to numpy arrays
    scores = np.array(scores)
    labels = np.array(labels)

    # Calculate precision, recall, and thresholds for PR curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    # Calculate ROC AUC and curve
    roc_auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    # Plot PR and ROC curves
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f'PR Curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random predictions curve
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()

    # Find the best threshold based on geometric mean of precision and recall
    gmean = (precision * recall)**0.5
    best_idx = np.argmax(gmean)
    best_threshold = thresholds[best_idx]

    # Make predictions based on the best threshold
    predictions = (scores > best_threshold).astype(int)

    # Calculate F1 score, accuracy, and confusion matrix
    f1 = f1_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    conf_matrix = confusion_matrix(labels, predictions)

    # Print classification report, accuracy, and confusion matrix
    print(classification_report(labels, predictions))
    print("Accuracy: ", accuracy)
    print("Confusion Matrix: \n", conf_matrix)
    print(f"PR AUC: {pr_auc:.4f}, ROC AUC: {roc_auc:.4f}, Best Threshold: {best_threshold:.4f}")

    return precision, recall, pr_auc, f1, best_threshold, accuracy, conf_matrix, roc_auc

# Load the Pre-trained IntroVAE Model
# Initializes the IntroVAE model structure and loads weights from a saved state dict,
# preparing the model for evaluation or further training.

# Function to load the model
def load_model(model_path, input_dim, latent_dim):
    # Initialize the model structure
    model = IntroVAE(input_dim=input_dim, latent_dims=latent_dim)
    # Load the saved weights
    model.load_state_dict(torch.load(model_path))
    # Set the model to evaluation mode
    model.eval()  
    return model

# Prepare Credit Card Transaction Data for Modeling
# Loads transaction data, applies standard scaling, and splits into training and testing sets.
# This standardized preprocessing pipeline ensures data is appropriately formatted for model input.

# Function to prepare the data
def prepare_data(dataset_path, test_size=0.2, random_state=42):
    # Load the dataset
    df = pd.read_csv(dataset_path)
    # Separate the features and the target
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalize features
    scaler = StandardScalerTransform()
    scaler.fit(X_train)  # Fit the scaler only on the training data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Generate Predictions and Evaluate Model on Test Data
# Utilizes the IntroVAE model to predict labels for given data and evaluates performance,
# returning a threshold for classifying transactions based on error reconstruction.

def predict_and_evaluate(model, X, y, batch_size=32):
    dataset = CreditCardDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _, _, _, _, best_threshold, _, _, _ = evaluate_model(model, dataloader, device)

    # Generate predictions
    scores = []
    with torch.no_grad():
        for inputs, _, _ in dataloader:
            inputs = inputs.to(device)
            reconstruction, _, _ = model(inputs)
            error = torch.mean((inputs - reconstruction) ** 2, dim=1)
            scores.extend(error.cpu().numpy())
    scores = np.array(scores)
    predictions = (scores > best_threshold).astype(int)

    return predictions

# Training Supervised Model with Pseudo Labels from IntroVAE
# Applies pseudo labels generated by IntroVAE as targets for training an XGBoost model,
# aiming to improve fraud detection performance by leveraging unsupervised learning insights.

# Define paths
best_model_path = 'best_model-elliptic_1000.pth'
dataset_path = 'dataset/creditcard.csv'

# Assuming these were the hyperparameters used for the model
input_dim = 30  # Number of features
latent_dim = 12

# Load the models
best_model = load_model(best_model_path, input_dim, latent_dim)

# Prepare the data
X_train, X_test, y_train, y_test = prepare_data(dataset_path)

# Generate predictions from the best model on the training data only
y_pred_train = predict_and_evaluate(best_model, X_train, y_train)

# Replace the training labels with the predictions from the unsupervised model
y_train = y_pred_train

# Prepare DMatrix for XGBoost Training
# Converts training and testing sets into DMatrix format, optimizing data structure for XGBoost training efficiency.

# Convert the dataset into an optimized data structure called Dmatrix that XGBoost supports
D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

# Hyperparameter Optimization using Grid Search
# Employs GridSearchCV to find the optimal set of parameters for the XGBoost model,
# based on Precision-Recall Area Under Curve (PR AUC) to focus on the model's performance in fraud detection.

scale_pos_weight = sum(1 - y_train) / sum(y_train) * 2

param = {
    'eta': [0.05],  
    'max_depth': [8], 
    'objective': ['binary:logistic'],
    'gamma': [0.2],  
    'subsample': [0.9], 
    'colsample_bytree': [0.9], 
    'lambda': [1.5],  
    'scale_pos_weight': [scale_pos_weight], 
    'min_child_weight': [3],
    'max_delta_step': [1]  
}

# Define a function to calculate PR AUC
def pr_auc_score(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

# Make a custom scorer using this function
pr_auc_scorer = make_scorer(pr_auc_score, response_method='predict_proba')

# Define the model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='aucpr')
# Perform Grid Search
grid_search = GridSearchCV(model, param, cv=5, scoring=pr_auc_scorer)
grid_search.fit(X_train, y_train)

# Display Comprehensive Evaluation Metrics
# Reports various metrics including accuracy, recall, precision, F1 score, PR AUC, and ROC AUC,
# offering a multi-faceted view of the model's fraud detection capabilities.

# Print the grid search settings
print("Grid Search settings:")
print(f"Model: {model}")
print(f"Parameters: {param}")
print(f"Cross-validation folds: 5")
print(f"Scoring metric: {pr_auc_scorer}")

# Get the best parameters
best_params = grid_search.best_params_

# Print the best parameters
print("Best parameters:")
print(best_params)

# Train the model with the best parameters
model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='aucpr')
model.fit(X_train, y_train)

# Predict the labels of the test set
preds = model.predict(X_test)


#print many other metrics
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))
# Print the accuracy
print("Accuracy = {}".format(accuracy_score(y_test, preds)))
#print recall
print("Recall = {}".format(recall_score(y_test, preds)))
#print f1
print("F1 = {}".format(f1_score(y_test, preds)))
#print precision
print("Precision = {}".format(precision_score(y_test, preds)))
#prin pr auc
precision, recall, thresholds = precision_recall_curve(y_test, preds)
pr_auc = auc(recall, precision)
print(f"PR AUC: {pr_auc:.4f}")
#ROC AUC SCORE
print(f"ROC AUC: {roc_auc_score(y_test, preds):.4f}")

# Print the best score and corresponding fold
print("Best score (PR AUC) from Grid Search:")
print(grid_search.best_score_)
print("Corresponding fold:")
print(grid_search.best_index_ + 1)


# Select and Save the Best Model from Grid Search
# Extracts the best performing model based on PR AUC and saves it for future prediction tasks,
# ensuring that the most effective model is readily available for deployment in fraud detection applications.

# Save the best model
best_model = grid_search.best_estimator_

# Save model to file
pickle.dump(best_model, open("xgboost_best_model_final.pkl", "wb"))

best_model.save_model("xgboost_best_model_final.json")