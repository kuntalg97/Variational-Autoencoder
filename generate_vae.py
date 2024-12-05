# Kuntal Ghosh
# November 2024

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from ase.io import read, write
import matplotlib.pyplot as plt

# Read the XYZ file
original_trajectory = read('original_configurations.xyz', index=':')

# Extract coordinates and create reaction coordinates
num_original_configs = len(original_trajectory)
num_atoms = len(original_trajectory[0])
original_configs = np.array([atoms.get_positions().flatten() for atoms in original_trajectory])
original_reaction_coords = np.linspace(0, 1, num_original_configs)

# Convert to PyTorch tensors
configs_tensor = torch.FloatTensor(original_configs)
reaction_coords_tensor = torch.FloatTensor(original_reaction_coords).unsqueeze(1)

# Combine configurations with reaction coordinates
input_data = torch.cat([configs_tensor, reaction_coords_tensor], dim=1)

# Print shapes for debugging
print(f"configs_tensor shape: {configs_tensor.shape}")
print(f"reaction_coords_tensor shape: {reaction_coords_tensor.shape}")
print(f"input_data shape: {input_data.shape}")

# Create DataLoader
dataset = TensorDataset(input_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim - 1, 128),  # -1 because there is no need to encode the reaction coordinate
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),  # +1 for reaction coordinate
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim - 1)  # -1 because we don't reconstruct the reaction coordinate
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, rc):
        z_rc = torch.cat([z, rc.unsqueeze(1)], dim=1)
        return self.decoder(z_rc)
    
    def forward(self, x):
        mu, logvar = self.encode(x[:, :-1])  # Don't encode the reaction coordinate
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x[:, -1]), mu, logvar

# Initialize the model
input_dim = input_data.shape[1]
latent_dim = 16
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        x = batch[0]
        recon_x, mu, logvar = model(x)
        recon_loss = nn.MSELoss()(recon_x, x[:, :-1])
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.1 * kl_div
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate new configurations
num_new_configs = 10000
model.eval()
with torch.no_grad():
    new_rcs = torch.linspace(0, 1, num_new_configs)
    z = torch.randn(num_new_configs, latent_dim)
    new_configs = model.decode(z, new_rcs)

# Convert to numpy and reshape
new_configs_np = new_configs.numpy().reshape(num_new_configs, num_atoms, 3)

# Create new Atoms objects for the generated configurations
new_trajectory = [original_trajectory[0].copy() for _ in range(num_new_configs)]
for i, config in enumerate(new_configs_np):
    new_trajectory[i].set_positions(config)

# Write the new trajectory to an XYZ file
write('new_configurations_vae.xyz', new_trajectory)

print(f"Generated {num_new_configs} new configurations and saved to 'new_configurations.xyz'")

# Visualize the results for the first atom's x-coordinate
plt.figure(figsize=(10, 6))
plt.scatter(original_reaction_coords, original_configs[:, 0], 
            color='red', label='Original')
plt.scatter(new_rcs.numpy(), new_configs_np[:, 0, 0], 
            color='blue', alpha=0.5, label='Generated')
plt.title("Generated vs Original - First Atom's X-coordinate")
plt.xlabel("Reaction Coordinate")
plt.ylabel("X-coordinate")
plt.legend()
plt.show()
