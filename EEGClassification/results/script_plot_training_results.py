import torch
import matplotlib.pyplot as plt
import os

# Names of the .pth files for different models
model_names = ['LSTM_embed_128_batch64_55_95', 'EEGNet_embed_128_batch64_55_95', 'EEGChannelNet_embed_128_batch64_55_95', 'EEG2Image_efficientnet_v2_s_5_95','EEG2Image_efficientnet_v2_s_5_95_canny_50_120']
modified_model_names = ['LSTM', 'EEGNet', 'EEGChannelNet', 'ICWA', 'ICWA+ED']
model_paths = [f"training_data_results/{model}.pth" for model in model_names]
# model_files = [model.split(".")[0] for model in model_files]

# Initialize lists to store validation loss and accuracy
val_losses = []
val_accuracies = []

# Load data from each file
for file in model_paths:
    data = torch.load(file)
    epochs_data = data['epochs']

    # Extract (100 EPOCHS) validation loss and accuracy for each epoch
    model_val_losses = [epoch['validation_loss'] for epoch in epochs_data][:100]
    model_val_accuracies = [epoch['validation_accuracy'] for epoch in epochs_data][:100]

    # Append to the lists
    val_losses.append(model_val_losses)
    val_accuracies.append(model_val_accuracies)

# Create figure for Validation Loss Plot
plt.figure(figsize=(6, 4))
for i, model_loss in enumerate(val_losses):
    plt.plot(model_loss, label=f'{modified_model_names[i]}')
plt.title('Validation Loss of Different Models (100 epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(fontsize='small')
plt.savefig('validation_loss_plot.pdf', format = 'pdf', dpi = 300)  # Save the figure
plt.close()  # Close the plot to avoid displaying it

# Create figure for Validation Accuracy Plot
plt.figure(figsize=(6, 4))
for i, model_acc in enumerate(val_accuracies):
    plt.plot(model_acc, label=f'{modified_model_names[i]}')
plt.title('Validation Accuracy of Different Models (100 epochs)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(fontsize='small')
plt.savefig('validation_accuracy_plot.pdf', format = 'pdf', dpi = 300)  # Save the figure
plt.close()  # Close the plot to avoid displaying it

print("Figures saved as 'validation_loss_plot.png' and 'validation_accuracy_plot.png'")
