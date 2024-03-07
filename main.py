"""
Credit Card Fraud Detection using IntroVAE

This Python module adapts concepts from the IntroVAE model originally developed for KPI anomaly detection 
(found at https://github.com/lyxiao15/SaVAE-SR/blob/main/main.py) to address the challenge of detecting credit card fraud.
While leveraging the foundational principles of VAE for anomaly detection, this implementation incorporates specific 
adaptations such as Elliptic Envelope for outlier detection, specialized preprocessing for credit card transaction data, 
and a tailored evaluation framework. The module demonstrates a novel application of IntroVAE models, extending its utility 
beyond time-series KPI data to financial fraud detection, thus showcasing the flexibility and adaptability of VAE-based models 
for varied applications.
"""


# Import necessary libraries
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.model_selection import StratifiedKFold
from source import IntroVAE, CreditCardDataset, StandardScalerTransform
from torch.utils.data import DataLoader
import zipfile
import os
from torch import device
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.covariance import EllipticEnvelope


# Set the device to GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the path of the zipped dataset and the path to extract it to
zip_path = 'dataset/fraud_credit.zip'
extract_to = 'dataset/'

# Check if the dataset is already unzipped
if not os.path.exists(extract_to + 'creditcard.csv'):
    # If not, unzip the dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Dataset unzipped successfully.")
else:
    print("Dataset already unzipped.")


# Argument Parser Function
# Configures and parses command-line options for model training and evaluation settings.
# Adapted from the original setup in SaVAE-SR to fit the specific requirements and parameters
# of the credit card fraud detection task, including data paths, model hyperparameters, and evaluation metrics.

# Function to parse command line arguments
def arg_parse():
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection')
    parser.add_argument("--data", dest='data_path', type=str, default='./dataset/creditcard.csv', help='The dataset path')
    parser.add_argument("--epochs", dest='epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", dest='batch_size', type=int, default=32, help="Batch size")
    parser.add_argument("--latent-dim", dest='latent_dim', type=int, default=20, help="Dimension of latent space")
    parser.add_argument("--learning-rate", dest='learning_rate', type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--results", dest='results_path', type=str, default='./results/results.csv', help="Path to save results")
    return parser.parse_args()


# Model Evaluation Function
# Assesses the performance of the IntroVAE model on a given dataset, calculating various metrics
# such as precision, recall, F1-score, and area under the precision-recall curve (PR AUC).
# This function extends the evaluation approach from SaVAE-SR with additional metrics relevant to 
# fraud detection and employs a dynamic thresholding technique to optimize for the best F1-score.

# Function to evaluate the model
def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    labels, scores = [], []
    with torch.no_grad():
        for inputs, targets, anomaly_scores in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move to the correct device
            reconstruction, _, _ = model(inputs)
            error = torch.mean((inputs - reconstruction) ** 2, dim=1)
            scores.extend(error.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    # Calculate dynamic threshold and metrics
    scores = np.array(scores)
    labels = np.array(labels)

    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # Calculate geometric mean for each threshold
    gmean_scores = np.sqrt(precision * recall)

    # Find the index of the maximum geometric mean
    best_index = np.argmax(gmean_scores)

    # Best threshold corresponding to the highest geometric mean
    best_threshold = thresholds[best_index]

    # Calculate predictions based on the best threshold
    predictions = (scores > best_threshold).astype(int)

    # Recalculate precision and recall for the best threshold for reporting
    pr_auc = auc(recall, precision)
    f1 = f1_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    conf_matrix = confusion_matrix(labels, predictions)

    # Print classification report, accuracy and confusion matrix
    print(classification_report(labels, predictions))
    print("Accuracy: ", accuracy)
    print("Confusion Matrix: \n", conf_matrix)

    return precision, recall, pr_auc, f1, best_threshold, accuracy, conf_matrix


# Results Saving Function
# Saves the evaluation metrics to a specified file path, ensuring results are preserved for further analysis.
# This utility function supports documentation of model performance across multiple runs, aiding in the iterative
# refinement of the fraud detection model. It is designed to complement the evaluation process by providing
# a straightforward mechanism for result storage.

# Function to save the results to a file
def save_results(results_path, results):
    results_dir = os.path.dirname(results_path)
    # Create the directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Write the results to the file
    with open(results_path, 'w') as file:
        file.write("PR AUC, F1 Score, Precision, Recall\n")
        for result in results:
            file.write(f"{result['pr_auc']}, {result['f1']}, {np.mean(result['precision'])}, {np.mean(result['recall'])}\n")


# Elliptic Envelope Anomaly Detection Function
# Applies the Elliptic Envelope method for identifying outliers within the credit card transaction data.
# This additional preprocessing step enhances the model's ability to recognize anomalous patterns, 
# directly addressing the challenge of unsupervised fraud detection. This function represents a novel integration 
# within the context of using VAE for fraud detection, demonstrating an innovative approach to improving model sensitivity.

# Function to apply Elliptic Envelope for anomaly detection
def apply_elliptic_envelope(X_train, expected_contamination=0.005, support_fraction=0.99):
    """
    Apply Elliptic Envelope to detect anomalies in the training data.
    
    Parameters:
    - X_train: Training data features.
    - expected_contamination: The proportion of outliers in the data set.
    - support_fraction: The proportion of points to be included in the minimum covariance determinant estimate.
    
    Returns:
    - scores: Anomaly scores for each instance in X_train.
    - labels: Binary labels for outliers (1) and inliers (0) based on the expected_contamination.
    """
    # Initialize the Elliptic Envelope model
    elliptic_env = EllipticEnvelope(contamination=expected_contamination, support_fraction=support_fraction)
    
    # Fit the model on the training data
    elliptic_env.fit(X_train)
    
    # Get the anomaly scores for the training data
    scores = elliptic_env.decision_function(X_train)
    
    # Get the binary labels (0: inliers, 1: outliers)
    labels = elliptic_env.predict(X_train)
    
    # Convert the prediction labels to match the original labels (0: normal, 1: fraud)
    labels[labels == 1] = 0
    labels[labels == -1] = 1
    
    return scores, labels

# Cross-Validation Function for Model Evaluation
# Implements a stratified k-fold cross-validation process to rigorously assess the model's performance
# across various splits of the dataset. This function ensures robust evaluation by leveraging multiple
# metrics and supporting model tuning. Adapted to the fraud detection domain, it highlights the model's 
# generalizability and effectiveness in detecting fraudulent transactions.

# Function to perform cross-validation on the model
def cross_validate_model(X, y, model_class, latent_dim, learning_rate, epochs, batch_size, folds=5):
    # Initialize StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=50)
    results = []
    best_pr_auc = 0
    best_model_state = None

    # Loop over each fold
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold}/{folds}")

        # Split the data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Normalize features
        scaler = StandardScalerTransform()
        scaler.fit(X_train)  # Fit the scaler on the training data
        X_train_scaled = scaler.transform(X_train)
        anomaly_scores_train, _ = apply_elliptic_envelope(X_train_scaled)
        X_test_scaled = scaler.transform(X_test)

        # Convert to PyTorch datasets, now including anomaly scores for training data
        train_dataset = CreditCardDataset(X_train_scaled, y_train, anomaly_scores_train)
        # For the test set, we do not include anomaly scores
        test_dataset = CreditCardDataset(X_test_scaled, y_test)

        # DataLoader creation remains the same
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = model_class(input_dim=X_train_scaled.shape[1], latent_dims=latent_dim, max_epochs=epochs, batch_size=batch_size, learning_rate_VAE=learning_rate).to(device)
        
        # Fit the model on the training data
        model.fit(train_dataloader)

        # Evaluate the model
        precision, recall, pr_auc, f1, best_threshold, accuracy, conf_matrix = evaluate_model(model, test_dataloader, device)

        # Save the best model state
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_state = model.state_dict()  # Save the best model state

        # Calculate TP, FP, TN, FN
        true_positives = conf_matrix[1, 1]
        false_positives = conf_matrix[0, 1]
        true_negatives = conf_matrix[0, 0]
        false_negatives = conf_matrix[1, 0]

        # Store results for the fold
        results.append({
            'fold': fold,
            'precision': precision.mean(),
            'recall': recall.mean(),
            'pr_auc': pr_auc,
            'f1': f1,
            'best_threshold': best_threshold,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        })

    # After all folds
    if best_model_state is not None:
        # Save the best model
        torch.save(best_model_state, 'best_model-elliptic_lat20.pth')
        print(f"Saved the best model with PR AUC: {best_pr_auc}")

    # Train a final model on the entire dataset using the given hyperparameters
    final_model = model_class(input_dim=X.shape[1], latent_dims=latent_dim, max_epochs=epochs, batch_size=batch_size, learning_rate_VAE=learning_rate).to(device)
    final_model.fit(DataLoader(CreditCardDataset(X, y), batch_size=batch_size, shuffle=True))

    # Create the output directory if it doesn't exist
    if not os.path.exists('final_model'):
        os.makedirs('final_model')

    # Save the final model
    torch.save(final_model.state_dict(), 'final_model/final_model-elliptic_lat20.pth')
    print("Saved the final model trained on the entire dataset.")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Transpose the DataFrame so that each fold becomes a column
    results_df = results_df.transpose()

    # Rename the columns to 'Fold 1', 'Fold 2', etc.
    results_df.columns = ['Fold ' + str(i+1) for i in range(results_df.shape[1])]

    # Print the metrics for each fold
    print(f"\nMetrics for each fold:\n{results_df}")

    # Optionally, save the results to a CSV
    pd.DataFrame(results).to_csv('cross_validation_results_lat20.csv', index=False)

    return results


# Main Function to Load Data and Perform Model Evaluation
# Orchestrates the loading of credit card transaction data, model training, evaluation, and saving of results.
# Tailored to the fraud detection use case, it integrates data preprocessing, cross-validation, and result reporting,
# showcasing the comprehensive workflow from data preparation to model assessment and optimization in the context 
# of credit card fraud detection.

# Main function to load data and perform cross-validation
def main():
    args = arg_parse()

    # Load dataset
    dataset_path = args.data_path
    df = pd.read_csv(dataset_path)
    X = df.drop('Class', axis=1).values 
    y = df['Class'].values

    # Perform cross-validation
    results = cross_validate_model(X, y, IntroVAE, args.latent_dim, args.learning_rate, args.epochs, args.batch_size, folds=5)
    
    # Calculate and print average results
    total_true_positives = sum([result['true_positives'] for result in results])
    total_false_positives = sum([result['false_positives'] for result in results])
    total_true_negatives = sum([result['true_negatives'] for result in results])
    total_false_negatives = sum([result['false_negatives'] for result in results])

    avg_precision = total_true_positives / (total_true_positives + total_false_positives)
    avg_recall = total_true_positives / (total_true_positives + total_false_negatives)
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    avg_accuracy = (total_true_positives + total_true_negatives) / (total_true_positives + total_false_positives + total_true_negatives + total_false_negatives)
    avg_pr_auc = np.mean([result['pr_auc'] for result in results])

    print("\nCross-Validation Results:")
    print(f"Average PR AUC: {avg_pr_auc}")
    print(f"Average F1 Score: {avg_f1}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average Accuracy: {avg_accuracy}")
    print(f"Best Threshold: {np.mean([result['best_threshold'] for result in results])}")
    # Save the cross-validation results
    save_results(args.results_path, results)

# Run the main function 
if __name__ == "__main__":
    main()