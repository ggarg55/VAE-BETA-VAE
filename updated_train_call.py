model_beta1 = VAE(IMAGE_CHANNELS, INITIAL_FILTERS, LATENT_DIM, FINAL_SPATIAL_SIZE).to(DEVICE)
optimizer = optim.Adam(model_beta1.parameters(), lr=LEARNING_RATE)

train_losses_beta1 = []
recon_losses_beta1 = []
kld_losses_beta1 = []

print("Starting Baseline Training (Beta=1.0)...")
for epoch in range(EPOCHS):
    avg_loss, avg_r_loss, avg_k_loss = train_one_epoch(model_beta1, train_loader, optimizer, beta=1.0)
    
    train_losses_beta1.append(avg_loss)
    recon_losses_beta1.append(avg_r_loss)
    kld_losses_beta1.append(avg_k_loss)
    
    print(f'Epoch {epoch+1}: Total Loss: {avg_loss:.4f} | Recon Loss: {avg_r_loss:.4f} | KL Loss: {avg_k_loss:.4f}')
