import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from BEATs import BEATs, BEATsConfig
from transformers import TrainingArguments, Trainer
import pandas as pd

from tqdm import tqdm
import librosa
import numpy as np

import pdb

torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        
        encoding = torch.tensor(self.encodings[idx])
        label = self.labels_to_num(self.labels[idx])
        item = {'encoding': encoding, 'category': label}
        
        return item

    def __len__(self):
        return len(self.encodings)
    
    def labels_to_num(self, cat):
#         esc50_labels = ['dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm',
#            'door_wood_knock', 'can_opening', 'crow', 'clapping', 'fireworks',
#            'chainsaw', 'airplane', 'mouse_click', 'pouring_water', 'train',
#            'sheep', 'water_drops', 'church_bells', 'clock_alarm',
#            'keyboard_typing', 'wind', 'footsteps', 'frog', 'cow',
#            'brushing_teeth', 'car_horn', 'crackling_fire', 'helicopter',
#            'drinking_sipping', 'rain', 'insects', 'laughing', 'hen', 'engine',
#            'breathing', 'crying_baby', 'hand_saw', 'coughing',
#            'glass_breaking', 'snoring', 'toilet_flush', 'pig',
#            'washing_machine', 'clock_tick', 'sneezing', 'rooster',
#            'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets']
        esc50_labels = ['clean', 'protest', 'smoke']

        labels_to_num = {}

        for i in range(len(esc50_labels)):
            labels_to_num[esc50_labels[i]] = i

        return labels_to_num[cat]

class BEATsFineTuning(object):
    
    def __init__(self, model_filename = 'fine-tuned-cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt', **kwargs):
     
        self.checkpoint = torch.load(f'models/{model_filename}')
        cfg = BEATsConfig(self.checkpoint['cfg'])
        self.pretrained_model = BEATs(cfg)
        
        self.pretrained_model.load_state_dict(self.checkpoint['model'])
        self.pretrained_model.train()
    
    def preprocess_audio(self, input_dir, files, SAMPLE_RATE = 16000, batch_size = 16, **kwargs):
                     
        tracks = list()
                     
        for idx, file in enumerate(tqdm(files)):

            track, _ = librosa.load(f'{input_dir}/{file}', sr=SAMPLE_RATE, dtype=np.float32)
            track = torch.from_numpy(track)

            tracks.append(track)
            
            if not idx % batch_size:

                #maxtrack =  max([ta.shape[-1] for ta in tracks])
                maxtrack = 160125

                padded = [torch.nn.functional.pad(torch.from_numpy(np.array(ta)),(0,maxtrack-ta.shape[-1])) for ta in tracks]

                audio = torch.stack(padded)
                
                yield audio 
                    
                tracks = list()
                
    def training(self, dataset, batch_size, **kwargs):
        # Configuration options
        k_folds = 5
        num_epochs = 30
        loss_function = nn.CrossEntropyLoss()

        # For fold results
        results = {}

        # Set fixed random number seed
        torch.manual_seed(42)

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=k_folds, shuffle=True)

        # Start print
        print('--------------------------------')

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

            # Print
            print(f'FOLD {fold}')
            print('--------------------------------')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                              dataset, 
                              batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                              dataset,
                              batch_size=batch_size, sampler=test_subsampler)

            model = self.pretrained_model
    
            for name, param in model.named_parameters():
                if 'predictor' not in name:   
                    param.requires_grad = False
                else:
                    param.requires_grad=True

            model.predictor = nn.Sequential(nn.LayerNorm((768, ), eps=1e-12), 
                                             nn.Linear(768, 3))

            model.train()

            # Initialize optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Run the training loop for defined number of epochs
            for epoch in range(0, num_epochs):

                running_loss = 0.0
                running_corrects = 0

                # Print epoch
                print(f'Starting epoch {epoch+1}')

                # Set current loss value
                current_loss = 0.0

                # Iterate over the DataLoader for training data
                for data in trainloader:

                    # Get inputs
                    inputs = data['encoding']
                    targets = data['category']

                    inputs = torch.tensor(inputs)
                    targets = torch.tensor(targets)


                    optimizer.zero_grad()

                    # Perform forward pass

                    padding_mask = torch.zeros(len(inputs), inputs.shape[1]).bool().to('cuda')
                    outputs = model.extract_features(inputs, padding_mask)[0]


                    # Compute loss
                    loss = loss_function(outputs, targets)

                    _, preds = torch.max(outputs, 1)

                    # Perform backward pass
                    loss.backward()

                    # Perform optimization
                    optimizer.step()

                    # Print statistics
                    running_loss += loss.item()*inputs.size(0)

                    #pdb.set_trace()
                    running_corrects += torch.sum(preds==targets)

                epoch_loss = running_loss / len(trainloader.dataset)
                epoch_acc = running_corrects.double() / len(trainloader.dataset)
                print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # Process is complete.
            print('Training process has finished. Saving trained model.')

            # Print about testing
            print('Starting testing')

            # Saving the model
            save_path = f'./model-fold-{fold}.pth'
            torch.save(model.state_dict(), save_path)

            # Evaluationfor this fold
            correct, total = 0, 0
            with torch.no_grad():

                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):

                    # Get inputs
                    inputs = data['encoding']
                    targets = data['category']
                    model.eval()

                    # Generate outputs

                    padding_mask = torch.zeros(len(inputs), inputs.shape[1]).bool().to('cuda')
                    outputs = model.extract_features(inputs, padding_mask)[0]

                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)

                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                # Print accuracy
                print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
                print('--------------------------------')
                results[fold] = 100.0 * (correct / total)

      # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
            print(f'Average: {sum/len(results.items())} %')
    
    def __call__(self, input_dir='/home/ubuntu/data/ESC-50-master/audio/', labels=None, **kwargs):

        if labels is None:
            labels = "/home/ubuntu/data/ESC-50-master/meta/esc50.csv"
        
        labels = pd.read_csv(labels)
        labels = labels.sample(frac=1).reset_index(drop=True)
            
        audio_dir = input_dir


        # ----- 1. Preprocess data -----#
        # Preprocess data
        X = list(labels["filename"])
        y = list(labels["category"])

        features = []

        for audio in self.preprocess_audio(audio_dir, X, **kwargs):
            features.extend(torch.Tensor(audio))
            torch.cuda.empty_cache()
        
        dataset = Dataset(features, y)
        self.training(dataset, **kwargs)
        
  
if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=1 << 6, help='batch size')
    parser.add_argument('-f','--input_dir')
    parser.add_argument('-l','--labels')
    parser.add_argument('-e', '--num_epochs', type=int, default=30)
    parser.add_argument('-m', '--model', default = 'models/fine-tuned-cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
    help='pre-trained model')
    args = parser.parse_args()

    self = BEATsFineTuning()
    self(**vars(args))