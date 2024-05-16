import torch 
import random
import copy

class EEGDataset:
    # Constructor
    def __init__(self, opt, eeg_signals_path):
        """
        opt: {
            subject:,
            time_low:,
            time_high:,
            model_type:
        }
        """
        self.opt = opt
        # Load EEG signals
        self.data = torch.load(eeg_signals_path)
        
        # self.data = [loaded['eeg'][i] for i in range(len(loaded['eeg']))]
        # else:
        #     self.data=loaded['dataset']        
        # self.labels = loaded["labels"]
        # self.images = loaded["images"]
        
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = torch.tensor(self.data[i]["eeg"], dtype=torch.float32)
        eeg = eeg.float().t() #convert from (128, 500) to (500, 128)
        eeg = eeg[self.opt.time_low:self.opt.time_high,:]

        if self.opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(128,self.opt.time_high-self.opt.time_low)
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label

class Splitter:
    def __init__(self, dataset, split_name):
        # Set EEG dataset
        self.dataset = dataset
        self.split_name = split_name
        indices = list(range(len(self.dataset)))
        random.shuffle(list(range(len(self.dataset))))

        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - (train_size + val_size)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]

        if self.split_name == 'train':
            self.split_idx =  train_indices
        elif self.split_name == 'val': 
            self.split_idx = val_indices
        else:
            self.split_idx = test_indices 
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label

