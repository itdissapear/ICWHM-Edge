import torch
import matplotlib.pyplot as plt
import os

# Names of the .pth files for different models
model_names = ['LSTM_embed_128_batch64_55_95', 'EEGNet_embed_128_batch64_55_95', 'EEGChannelNet_embed_128_batch64_55_95', 'EEG2Image_efficientnet_v2_s_5_95','EEG2Image_efficientnet_v2_s_5_95_canny_50_120']
modified_model_names = ['LSTM', 'EEGNet', 'EEGChannelNet', 'ICWA', 'ICWA+ED']
model_paths = [f"training_data_results/{model}.pth" for model in model_names]
# model_files = [model.split(".")[0] for model in model_files]

# Load data from each file
for i in range(len(model_paths)):
    data = torch.load(model_paths[i])
    
    print(f"{modified_model_names[i]} - Val acc: {data['best_val_accuracy']}; Test acc: {data['test_accuracy']}\n")


