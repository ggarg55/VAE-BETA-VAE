def plot_all_losses(total_losses, recon_losses, kld_losses, title="Training Losses"):
    plt.figure(figsize=(10, 5))
    plt.plot(total_losses, label="Total Loss", color='blue')
    plt.plot(recon_losses, label="Reconstruction Loss", color='green', linestyle='--')
    plt.plot(kld_losses, label="KL Divergence Loss", color='red', linestyle='--')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_all_losses(train_losses_beta1, recon_losses_beta1, kld_losses_beta1, title="Beta=1.0 Training Losses")
