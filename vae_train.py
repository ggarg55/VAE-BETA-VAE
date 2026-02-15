import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 128
INITIAL_FILTERS = 32
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 2 # Reduced for demonstration purposes. Use 70 for full training.
IMAGE_CHANNELS = 3
FINAL_SPATIAL_SIZE = 4 # Encoder output (4x4)

# Data Preparation
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Encoder
class Encoder(nn.Module):
    def __init__(self, image_channels, initial_filters, latent_dim):
        super(Encoder, self).__init__()
        self.image_channels = image_channels
        self.initial_filters = initial_filters
        self.latent_dim = latent_dim
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1
        self.flattened_size = 128 * 4 * 4

        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=self.image_channels, out_channels=self.initial_filters, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv2 = nn.Conv2d(in_channels=initial_filters, out_channels=initial_filters * 2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv3 = nn.Conv2d(in_channels=initial_filters*2, out_channels=initial_filters * 4, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, initial_filters, final_spatial_size):
        super(Decoder, self).__init__()
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1
        self.latent_dim = latent_dim
        self.initial_filters = initial_filters
        self.final_spatial_size = final_spatial_size
        self.start_filters = initial_filters * 4
        self.fc_input_dim = self.start_filters * final_spatial_size * final_spatial_size

        self.fc = nn.Linear(latent_dim, self.fc_input_dim)
        self.deconv1 = nn.ConvTranspose2d(self.start_filters, initial_filters * 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(initial_filters * 2, initial_filters, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(initial_filters, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.start_filters, self.final_spatial_size, self.final_spatial_size)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + (eps * std)
    return z

# VAE
class VAE(nn.Module):
    def __init__(self, image_channels, initial_filters, latent_dim, final_spatial_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(image_channels, initial_filters, latent_dim)
        self.decoder = Decoder(latent_dim, initial_filters, final_spatial_size)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparameterize(mu, log_var)
        recon_image = self.decoder(z)
        return recon_image, mu, log_var

def vae_loss_function(recon_x, x, mu, log_var, beta_value):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + (beta_value * kld_loss)
    return total_loss, recon_loss, kld_loss

def train_one_epoch(model, dataloader, optimizer, beta):
    model.train()
    total_train_loss = 0
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(DEVICE)
        optimizer.zero_grad()
        recon_images, mu, log_var = model(images)
        loss, r_loss, k_loss = vae_loss_function(recon_images, images, mu, log_var, beta)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss / len(dataloader.dataset)

def interpolate_latent_space(model, steps=10, filename=None):
    model.eval()
    with torch.no_grad():
        z1 = torch.randn(1, LATENT_DIM).to(DEVICE)
        z2 = torch.randn(1, LATENT_DIM).to(DEVICE)
        alphas = np.linspace(0, 1, steps)
        interpolated_images = []
        for alpha in alphas:
            z_new = (1 - alpha) * z1 + alpha * z2
            reconstructed_img = model.decoder(z_new)
            interpolated_images.append(reconstructed_img.squeeze().cpu())
        
        fig, axes = plt.subplots(1, steps, figsize=(20, 2))
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img.permute(1, 2, 0))
            axes[i].axis('off')
        if filename:
            plt.savefig(filename)
        plt.close()

def generate_image_grid(model, grid_size=4, filename=None):
    model.eval()
    with torch.no_grad():
        z = torch.randn(grid_size * grid_size, LATENT_DIM).to(DEVICE)
        generated_images = model.decoder(z).cpu()
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        for i in range(grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            axes[row, col].imshow(generated_images[i].permute(1, 2, 0))
            axes[row, col].axis('off')
        if filename:
            plt.savefig(filename)
        plt.close()

# Main execution
if __name__ == "__main__":
    
    # Train Beta = 1.0
    model_beta1 = VAE(IMAGE_CHANNELS, INITIAL_FILTERS, LATENT_DIM, FINAL_SPATIAL_SIZE).to(DEVICE)
    optimizer_beta1 = optim.Adam(model_beta1.parameters(), lr=LEARNING_RATE)
    train_losses_beta1 = []
    
    print("Starting Baseline Training (Beta=1.0)...")
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model_beta1, train_loader, optimizer_beta1, beta=1.0)
        train_losses_beta1.append(avg_loss)
        print(f'Epoch {epoch+1}: Average Loss: {avg_loss:.4f}')

    # Train Beta = 5.0
    model_beta5 = VAE(IMAGE_CHANNELS, INITIAL_FILTERS, LATENT_DIM, FINAL_SPATIAL_SIZE).to(DEVICE)
    optimizer_beta5 = optim.Adam(model_beta5.parameters(), lr=LEARNING_RATE)
    train_losses_beta5 = []

    print("\nStarting Experiment Training (Beta=5.0)...")
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model_beta5, train_loader, optimizer_beta5, beta=5.0)
        train_losses_beta5.append(avg_loss)
        print(f'Epoch {epoch+1}: Average Loss: {avg_loss:.4f}')

    # 1. Training Curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_beta1, label='Beta=1.0')
    plt.plot(train_losses_beta5, label='Beta=5.0')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves')
    plt.savefig('training_curves.png')
    plt.close()
    print("Saved training_curves.png")

    # 2. Generated Grid
    generate_image_grid(model_beta1, filename='grid_beta1.png')
    generate_image_grid(model_beta5, filename='grid_beta5.png')
    print("Saved grid_beta1.png and grid_beta5.png")

    # 3. Interpolation
    interpolate_latent_space(model_beta1, filename='interpolation_beta1.png')
    interpolate_latent_space(model_beta5, filename='interpolation_beta5.png')
    print("Saved interpolation_beta1.png and interpolation_beta5.png")