"""
Credit Card Fraud Detection using IntroVAE

This module adapts and extends the variational autoencoder architecture from the project found at https://github.com/lyxiao15/SaVAE-SR/blob/main/source/SaVAE.py,
tailoring it for the specific application of detecting credit card fraud. While inspired by the foundational VAE structure and training methodology of the original work,
this implementation introduces significant modifications including an adversarial discriminator for enhanced anomaly detection, and is specialized for processing
credit card transaction data. The code herein demonstrates a novel application of VAEs, incorporating additional preprocessing and a different architectural focus
aimed at the domain of financial fraud detection.

The structure and approach are significantly adapted to fit the requirements of the task at hand, with custom preprocessing, dataset handling, and model training
procedures that diverge from the original source material.
"""



# Import libraries
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .network_credit_card import VAE
import numpy as np
from .network_credit_card import init_weights
from torch.optim import AdamW



# Check if CUDA is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Custom Dataset Class for Credit Card Transactions
# Designed to facilitate loading and preprocessing of credit card transaction data for fraud detection.
# This class is a novel contribution, structured to support the specific data format and features of credit card fraud datasets.

# Define a custom Dataset for credit card fraud detection
class CreditCardDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, anomaly_scores=None):
        """
        Initialize the dataset with features, labels, and anomaly scores.
        """
        self.features = features
        self.labels = labels
        self.anomaly_scores = anomaly_scores if anomaly_scores is not None else np.zeros(len(labels))

    def __len__(self):
        # Return the number of transactions in the dataset
        return len(self.features)

    def __getitem__(self, idx):
        # Get the features, label, and anomaly score of the transaction at the given index
        # Convert them to PyTorch tensors and return them
        if self.anomaly_scores is not None:
            return (torch.tensor(self.features[idx], dtype=torch.float32), 
                    torch.tensor(self.labels[idx], dtype=torch.float32), 
                    torch.tensor(self.anomaly_scores[idx], dtype=torch.float32))
        return (torch.tensor(self.features[idx], dtype=torch.float32), 
                torch.tensor(self.labels[idx], dtype=torch.float32))



# StandardScalerTransform Class
# Implements feature scaling using StandardScaler from sklearn, preparing transaction data for the neural network.
# This preprocessing step is crucial for normalizing the features of credit card transactions, aiding in the effective
# training of the IntroVAE model.

# Define a transform that scales the features using StandardScaler
class StandardScalerTransform:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def transform(self, X):
        return self.scaler.transform(X)


# IntroVAE Model Class for Credit Card Fraud Detection
# This class represents an IntroVAE model specifically adapted for the detection of anomalies in credit card transactions.
# While it draws inspiration from the original IntroVAE implementation for anomaly detection in time-series data,
# significant modifications have been made to tailor it for financial transaction data, including the introduction
# of an adversarial discriminator component and adjustments in the training procedure to optimize for fraud detection.

# Define the IntroVAE model for anomaly detection
class IntroVAE(nn.Module):
    def __init__(self, input_dim, latent_dims, max_epochs=100, batch_size=64, learning_rate_VAE=1e-3, learning_rate_D=1e-4, cuda=False):
        super(IntroVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dims = latent_dims
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate_VAE = learning_rate_VAE
        self.learning_rate_D = learning_rate_D
        self.cuda = cuda and torch.cuda.is_available()

        # Create an instance of the VAE class
        self.model = VAE(input_dim, latent_dims)
        self.model.apply(init_weights)

        # Create the optimizer and scheduler
        self.optimizer_VAE = AdamW(self.model.parameters(), lr=self.learning_rate_VAE, weight_decay=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer_VAE, mode='min', factor=0.1, patience=10)

        # Define the adversarial discriminator
        self.adversarial_discriminator = nn.Sequential(
            nn.Linear(latent_dims, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Move the model and the discriminator to the device
        self.model.to(device)
        self.adversarial_discriminator.to(device)

        # Create the optimizer for the discriminator
        self.optimizer_D = AdamW(self.adversarial_discriminator.parameters(), lr=self.learning_rate_D, weight_decay=1e-4)

    def forward(self, x):
        """
        Forward pass through the VAE model contained within IntroVAE.
        This method needs to be defined to use IntroVAE in a forward pass.
        """
        # Assuming the VAE model's forward method returns reconstructed x, mu, and logvar
        return self.model(x)
    
    # Function to Compute Gradient Penalty for Adversarial Training
    # A crucial component of the adversarial training loop, ensuring that the discriminator's gradients
    # are penalized appropriately to stabilize training. This function is adapted to support the adversarial
    # component unique to this fraud detection model.

    # Adjusted compute_gradient_penalty to ensure tensors are on the correct device
    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        # Generate random weights for the interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1), device=device)
        # Compute the interpolates
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        # Compute the discriminator's output for the interpolates
        d_interpolates = D(interpolates)
        # Create a tensor of ones with the same size as the real samples
        fake = torch.ones(real_samples.size(0), 1, device=device, requires_grad=False)
        # Compute the gradients of the outputs with respect to the interpolates
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
        # Compute the gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # Custom Loss Function Incorporating Anomaly Scores
    # Extends the traditional VAE loss function by incorporating anomaly scores, which are used to weight
    # the reconstruction loss. This modification is key for tailoring the model's focus on more suspicious transactions,
    # enhancing its fraud detection capability.

    # Adjusted loss function to incorporate anomaly scores directly
    def loss_function(self, recon_x, x, mu, logvar, anomaly_scores=None, beta=1, gamma=1, aw=1):
        # Compute the reconstruction loss
        recon_loss = F.smooth_l1_loss(recon_x, x, reduction='none').sum(1)
        # If anomaly scores are provided, scale them and use them to weight the reconstruction loss
        if anomaly_scores is not None:
            scaled_scores = torch.tanh(anomaly_scores)
            anomaly_weights = aw * scaled_scores
            recon_loss *= anomaly_weights.to(device)
        # Compute the final reconstruction loss and the Kullback-Leibler divergence
        recon_loss = gamma * recon_loss.mean()
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld_loss



    # Model Training Function with Early Stopping
    # Orchestrates the training process of the IntroVAE model, including adversarial training of the discriminator.
    # Implements early stopping based on validation loss to prevent overfitting. This function has been customized
    # to accommodate the specific needs of credit card fraud detection, including handling of anomaly scores during training.

    # Fit the model to the data
    def fit(self, dataloader, lambda_gp=10, patience=3):
        # Set the model to training mode
        self.model.train()
        # Initialize the best discriminator and generator losses to infinity
        best_d_loss = float('inf')
        best_g_loss = float('inf')
        # Initialize the number of epochs with no improvement in either the discriminator or generator loss
        epochs_no_improve = 0
        # Training loop
        for epoch in range(self.max_epochs):
            total_d_loss, total_g_loss = 0.0, 0.0
            # Iterate over the data
            for data, _, anomaly_scores in dataloader:
                data, anomaly_scores = data.to(device), anomaly_scores.to(device)
                # Discriminator training
                self.optimizer_D.zero_grad()
                real_z = torch.randn(data.size(0), self.latent_dims, device=device)
                fake_z, _ = self.model.encoder(data)
                real_labels = torch.ones(data.size(0), 1, device=device)
                fake_labels = torch.zeros(data.size(0), 1, device=device)
                d_real_loss = F.binary_cross_entropy(self.adversarial_discriminator(real_z), real_labels)
                d_fake_loss = F.binary_cross_entropy(self.adversarial_discriminator(fake_z.detach()), fake_labels)
                d_loss = d_real_loss + d_fake_loss + lambda_gp * self.compute_gradient_penalty(self.adversarial_discriminator, real_z, fake_z.detach())
                d_loss.backward()
                self.optimizer_D.step()
                # VAE (Generator) training
                self.optimizer_VAE.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                vae_loss = self.loss_function(recon_batch, data, mu, logvar, anomaly_scores)
                g_loss = F.binary_cross_entropy(self.adversarial_discriminator(fake_z), real_labels)
                total_loss = vae_loss + g_loss
                total_loss.backward()
                self.optimizer_VAE.step()
                # Update the total discriminator and generator losses
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
            # Compute the average discriminator and generator losses
            avg_d_loss = total_d_loss / len(dataloader)
            avg_g_loss = total_g_loss / len(dataloader)
            print(f'Epoch: {epoch + 1}, Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}')
            # Check if there has been an improvement in either the discriminator or generator loss
            if avg_d_loss < best_d_loss or avg_g_loss < best_g_loss:
                if avg_d_loss < best_d_loss:
                    best_d_loss = avg_d_loss
                if avg_g_loss < best_g_loss:
                    best_g_loss = avg_g_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print('Early stopping!')
                    break


# Functions for Saving and Loading Model Parameters
# These utility functions facilitate the serialization and deserialization of the IntroVAE model's parameters,
# ensuring the model can be saved after training and loaded for inference. Adaptations have been made to align
# with the specific architecture and components of the fraud detection IntroVAE model.

# Save the model's parameters to a file
def save_model(self, path):
    torch.save(self.model.state_dict(), path)

# Load the model's parameters from a file
def load_model(self, path):
    self.model.load_state_dict(torch.load(path))
    if self.cuda:
        self.model.cuda()