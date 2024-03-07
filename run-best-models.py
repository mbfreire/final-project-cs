"""
Model Evaluation for Credit Card Fraud Detection Using IntroVAE

This module is dedicated to the comprehensive evaluation of two instances of an IntroVAE model — termed as the 'best model' 
and 'final model' — applied to the task of credit card fraud detection. The IntroVAE model, a Self-Adversarial Variational 
Autoencoder with Elliptic Convergence, is utilized here for its innovative approach in learning complex, high-dimensional 
data distributions for the purpose of anomaly (fraud) detection.

The evaluation encompasses a suite of metrics essential for assessing the performance of fraud detection systems, including:
- Precision: The ratio of correctly predicted positive observations to the total predicted positives, crucial for fraud detection to minimize false positives.
- Recall: The ratio of correctly predicted positive observations to all actual positives, vital for capturing as many fraudulent transactions as possible.
- F1 Score: The weighted average of Precision and Recall, providing a single metric to assess the balance between them.
- Accuracy: Overall, how often the model is correct, considering both fraud and non-fraud predictions.
- Confusion Matrix: A detailed breakdown of prediction outcomes, allowing for a nuanced understanding of model performance beyond aggregate metrics.
- PR AUC: The area under the Precision-Recall curve, indicating the trade-off between precision and recall for different probability thresholds.
- ROC AUC: The area under the Receiver Operating Characteristic curve, evaluating the model's ability to discriminate between classes.
- Best Threshold: The optimal cut-off point to classify transactions as fraudulent or legitimate, derived from maximizing the geometric mean of precision and recall.

These metrics collectively offer a view of the model's efficacy in identifying fraudulent activities within credit card transactions. By evaluating both the best and final models, this module aims to determine the most effective configuration for real-world application, balancing sensitivity to fraud with the practical need for precision.

"""
# Import necessary libraries
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from source import IntroVAE, CreditCardDataset, StandardScalerTransform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set the device to GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to evaluate the model
def evaluate_model(model, dataloader, device):
    # Set the model to evaluation mode
    model.eval()
    labels, scores = [], []
    # No gradient computation needed during evaluation
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            # Move inputs and targets to the device
            inputs, targets = inputs.to(device), targets.to(device)
            # Get the model's reconstruction of the inputs
            reconstruction, _, _ = model(inputs)
            # Calculate the reconstruction error
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

# Function to load the model from a given path
def load_model(model_path, input_dim, latent_dim):
    # Initialize the model structure
    model = IntroVAE(input_dim=input_dim, latent_dims=latent_dim)
    # Load the saved weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to prepare the data for model input
def prepare_data(dataset_path):
    # Load the dataset
    df = pd.read_csv(dataset_path)
    # Separate features and target
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # Normalize features
    scaler = StandardScalerTransform()
    scaler.fit(X)  # Fit the scaler on the entire data
    X_scaled = scaler.transform(X)

    return X_scaled, y

# Function to predict and evaluate the model
def predict_and_evaluate(model, X, y, batch_size=32):
    # Create a dataset from the data
    dataset = CreditCardDataset(X, y)
    # Create a dataloader for batch processing
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # Move the model to the device
    model.to(device)

    # Use existing evaluate_model function
    return evaluate_model(model, dataloader, device)

# Define paths
best_model_path = 'best_model-elliptic_1000.pth'
final_model_path = 'final_model/final_model-elliptic_1000.pth'
dataset_path = 'dataset/creditcard.csv'

# Assuming these were the hyperparameters used for the model
input_dim = 30
latent_dim = 12

# Load the models
best_model = load_model(best_model_path, input_dim, latent_dim)
final_model = load_model(final_model_path, input_dim, latent_dim)

# Prepare the data
X, y = prepare_data(dataset_path)

# Predict and evaluate with the best model
print("Evaluating the best fold model on the entire dataset:")
predict_and_evaluate(best_model, X, y)

# Predict and evaluate with the final model
print("\nEvaluating the final model on the entire dataset:")
predict_and_evaluate(final_model, X, y)