import numpy as np
import pandas as pd

import torch  
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader

# Use GPU if available  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VAE encoder
class Encoder(nn.Module):  
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):  
        super(Encoder, self).__init__()  
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) 

    def forward(self, X): 
        h = torch.relu(self.fc1(X)) 
        mu = self.fc_mu(h) 
        logvar = self.fc_logvar(h) 
        return mu, logvar

# VAE decoder
class Decoder(nn.Module):  
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784, likelihood="gaussian"):  
        super(Decoder, self).__init__()
        self.likelihood = likelihood
        self.fc1 = nn.Linear(latent_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        out = self.fc2(h)
        if self.likelihood == "poisson":
            return F.softplus(out) # rate λ > 0
        else:
            return out # Gaussian mean (real-valued)

# VAE class, combining encoder and decoder
class VAE(nn.Module):  
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, likelihood="gaussian"):  
        super(VAE, self).__init__()  
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)  
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, likelihood=likelihood)  
    
    def reparameterize(self, mu, logvar):  
        std = torch.exp(0.5 * logvar)  # Standard deviation  
        eps = torch.randn_like(std)    # Random noise  
        return mu + eps * std          # Sample from the latent distribution  
        
    def forward(self, X):  
        mu, logvar = self.encoder(X)  
        z = self.reparameterize(mu, logvar)  
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
