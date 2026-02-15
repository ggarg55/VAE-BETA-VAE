def generate_image_grid(model, grid_size=4):
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
        plt.show()

# Call the function for both models
print("Generated Grid for Beta=1.0:")
generate_image_grid(model_beta1)

print("Generated Grid for Beta=5.0:")
generate_image_grid(model_beta5)
