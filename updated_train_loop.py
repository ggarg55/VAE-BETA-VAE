def train_one_epoch(model, dataloader, optimizer, beta):
    model.train()
    total_train_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(DEVICE)
        optimizer.zero_grad()

        # Forward pass
        recon_images, mu, log_var = model(images)

        # Loss calculation
        loss, r_loss, k_loss = vae_loss_function(recon_images, images, mu, log_var, beta)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_recon_loss += r_loss.item()
        total_kld_loss += k_loss.item()

    num_samples = len(dataloader.dataset)
    avg_total_loss = total_train_loss / num_samples
    avg_recon_loss = total_recon_loss / num_samples
    avg_kld_loss = total_kld_loss / num_samples

    return avg_total_loss, avg_recon_loss, avg_kld_loss
