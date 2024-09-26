import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def normalize_data(data,min_val, max_val):
    return (data - min_val) / (max_val - min_val) * 2 - 1

def inverse_normalize(data, min_val, max_val):
    return (data + 1) / 2 * (max_val - min_val) + min_val

def inverse_normalize_old(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    return (data + 1) / 2 * (max_val - min_val) + min_val


# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, data_dim),
            nn.Identity() ##nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train(df,batch_size,latent_dim,num_epochs,lr):
    features = df.values # Load and preprocess the dataset
    tensor_features = torch.FloatTensor(features) # Convert data to PyTorch tensors
    

    min_val = torch.min(tensor_features)
    max_val = torch.max(tensor_features)
    
    normalized_data = normalize_data(tensor_features,min_val,max_val)
        
    dataset = TensorDataset(normalized_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    data_dim = normalized_data.shape[1]
    generator = Generator(latent_dim, data_dim)
    discriminator = Discriminator(data_dim)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.1)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.1)
    
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            real_data = batch[0]
            batch_size = real_data.size(0)

            optimizer_D.zero_grad()
    
            real_labels = torch.ones(batch_size, 1)
            real_outputs = discriminator(real_data)
            real_loss = criterion(real_outputs, real_labels)
    
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            fake_labels = torch.zeros(batch_size, 1)
            fake_outputs = discriminator(fake_data.detach())
            fake_loss = criterion(fake_outputs, fake_labels)
    
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()
    
            optimizer_G.zero_grad()
    
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            fake_outputs = discriminator(fake_data)
            g_loss = criterion(fake_outputs, real_labels)
    
            g_loss.backward()
            optimizer_G.step()
        scheduler_G.step()
        scheduler_D.step() 
    
        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    print('Training complete')
    return generator, min_val, max_val