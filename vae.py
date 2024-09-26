import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim)
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

def train(df,batch_size,latent_dim,num_epochs,lr):
    features = df.values
    tensor_features = torch.FloatTensor(features)
    
    dataset = TensorDataset(tensor_features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = tensor_features.shape[1]
    vae = VAE(input_dim, latent_dim)
    
    def loss_function(x_recon, x, mu, log_var):
        recon_loss = nn.L1Loss()(x_recon, x)  # Use L1 loss for reconstruction
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        #print(f'recon loss {recon_loss}, KL: {kl_div}')
        return recon_loss + 1.0 * kl_div  # Adjust the weight of the KL divergence term
    
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            x_recon, mu, log_var = vae(x)
            loss = loss_function(x_recon, x, mu, log_var)
            loss.backward()
            optimizer.step()
        scheduler.step()  # Update the lr
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    print('Training complete')
    return vae