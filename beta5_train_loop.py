model_beta5 = VAE(IMAGE_CHANNELS, INITIAL_FILTERS, LATENT_DIM, FINAL_SPATIAL_SIZE).to(DEVICE)
optimizer_beta5 = optim.Adam(model_beta5.parameters(), lr=LEARNING_RATE)

train_losses_beta5 = []
recon_losses_beta5 = []
kld_losses_beta5 = []

print("\nStarting Experiment Training (Beta=5.0)...")
for epoch in range(EPOCHS):
    avg_loss, avg_r_loss, avg_k_loss = train_one_epoch(model_beta5, train_loader, optimizer_beta5, beta=5.0)
    
    train_losses_beta5.append(avg_loss)
    recon_losses_beta5.append(avg_r_loss)
    kld_losses_beta5.append(avg_k_loss)
    
    print(f'Epoch {epoch+1}: Total Loss: {avg_loss:.4f} | Recon Loss: {avg_r_loss:.4f} | KL Loss: {avg_k_loss:.4f}')
