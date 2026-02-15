def compare_beta_curves(train_losses_beta1, train_losses_beta5):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_beta1, label='Beta=1.0', color='blue')
    plt.plot(train_losses_beta5, label='Beta=5.0', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title('Comparison of Training Loss Curves')
    plt.grid(True, alpha=0.3)
    plt.show()


compare_beta_curves(train_losses_beta1, train_losses_beta5)
