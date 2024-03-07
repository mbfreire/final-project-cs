# Import libraries
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# This CreditCardFraudDataset class is inspired by general data handling practices in PyTorch's Dataset usage as seen in various examples, 
# including the structure and approach for custom datasets demonstrated in the SaVAE-SR project (https://github.com/lyxiao15/SaVAE-SR/blob/main/source/data.py).
# Specific implementations, such as data loading from CSV, feature scaling, and the problem domain (credit card fraud detection), are original to this work.
# Custom Dataset for credit card fraud detection
class CreditCardFraudDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with transactions.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the data from the CSV file
        self.data_frame = pd.read_csv(csv_file)
        # Separate the features and labels
        self.features = self.data_frame.iloc[:, :-1].values
        self.labels = self.data_frame.iloc[:, -1].values
        # Scale the features to the range [0, 1]
        self.scaler = MinMaxScaler() 
        self.features = self.scaler.fit_transform(self.features) 
        # Store the transform function
        self.transform = transform

    def __len__(self):
        # Return the number of transactions in the dataset
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get the features and label of the transaction at the given index
        transaction = self.features[idx]
        label = self.labels[idx]
        # Create a sample dictionary
        sample = {'transaction': transaction, 'label': label}

        # Apply the transform function if it's provided
        if self.transform:
            sample = self.transform(sample)

        # Return the sample
        return sample


# The ToTensor transformation class follows a common pattern in PyTorch for converting NumPy arrays to PyTorch tensors, 
# akin to patterns seen in various PyTorch examples and the SaVAE-SR project.
    
# Define a transform that converts the features and label to PyTorch tensors
class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # Get the features and label from the sample
        transaction, label = sample['transaction'], sample['label']
        # Convert the features and label to PyTorch tensors and return them
        return {'transaction': torch.from_numpy(transaction).float(),
                'label': torch.tensor(label).float()}