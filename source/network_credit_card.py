# This implementation extends and modifies the initial VAE model structure provided in https://github.com/lyxiao15/SaVAE-SR/blob/main/source/network.py.
# Significant modifications include architecture adjustments, layer enhancements, and weight initialization.
# Original structure inspired by the provided code with references to the adaptations.

#improt libraries
import torch
import torch.nn as nn



# Modified from original Encoder class to include additional layers (BatchNorm1d, Dropout)
# and increased layer dimensions for complexity.
class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, eps=1e-4):
        super(Encoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.eps = eps
        self.layer1 = nn.Sequential(
            nn.Linear(x_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3)  # Slightly reduced dropout
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 512),  # Increased complexity
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.z_mean_layer = nn.Linear(512, z_dim)  # Adjusted for new layer size
        self.z_std_layer = nn.Sequential(
            nn.Linear(512, z_dim),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        z_mean = self.z_mean_layer(x)
        z_std = self.z_std_layer(x) + self.eps
        return z_mean, z_std

# Adapted from original Generator class with increased dimensions and added
# BatchNorm1d and Dropout for regularization and complexity enhancement.
class Generator(nn.Module):
    def __init__(self, z_dim, x_dim, eps=1e-4):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.eps = eps
        self.layer1 = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.x_mean_layer = nn.Linear(256, x_dim)
        self.x_std_layer = nn.Sequential(
            nn.Linear(256, x_dim),
            nn.Softplus()
        )

    def forward(self, z):
        z = self.layer1(z)
        z = self.layer2(z)
        x_mean = self.x_mean_layer(z)
        x_std = self.x_std_layer(z) + self.eps
        return x_mean, x_std


# The VAE class and reparameterization logic were retained from the original code,
# with modifications to integrate the redesigned Encoder and Generator classes.
class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, eps=1e-4, L=1):
        super(VAE, self).__init__()
        self.encoder = Encoder(x_dim=x_dim, z_dim=z_dim, eps=eps)
        self.generator = Generator(z_dim=z_dim, x_dim=x_dim, eps=eps)

    def forward(self, x):
        z_mean, z_std = self.encoder(x)
        z = self.reparameterization(z_mean, z_std)
        x_mean, x_std = self.generator(z)
        return x_mean, z_mean, z_std

    def reparameterization(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

# Apply weight initialization
# Weight initialization function for neural network layers, utilizing kaiming_uniform_
# for weight and setting biases, introduced to enhance model initialization.
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        m.bias.data.fill_(0.01)
