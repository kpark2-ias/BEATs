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
        esc50_labels = ['dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm',
           'door_wood_knock', 'can_opening', 'crow', 'clapping', 'fireworks',
           'chainsaw', 'airplane', 'mouse_click', 'pouring_water', 'train',
           'sheep', 'water_drops', 'church_bells', 'clock_alarm',
           'keyboard_typing', 'wind', 'footsteps', 'frog', 'cow',
           'brushing_teeth', 'car_horn', 'crackling_fire', 'helicopter',
           'drinking_sipping', 'rain', 'insects', 'laughing', 'hen', 'engine',
           'breathing', 'crying_baby', 'hand_saw', 'coughing',
           'glass_breaking', 'snoring', 'toilet_flush', 'pig',
           'washing_machine', 'clock_tick', 'sneezing', 'rooster',
           'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets', 'protest', 'smoke']
        #esc50_labels = ['clean', 'protest', 'smoke']

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
    
    def preprocess_audio(self, input_dir, files, SAMPLE_RATE = 44100, batch_size = 16, **kwargs):
                     
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
                
    def training(self, dataset, testset, data_type, batch_size, num_epochs, **kwargs):
        

        # Configuration options
        num_epochs = num_epochs
        loss_function = nn.CrossEntropyLoss()

        # For fold results
        results = {}

        # Set fixed random number seed
        torch.manual_seed(42)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                          dataset, 
                          batch_size=batch_size)

        model = self.pretrained_model

        for name, param in model.named_parameters():
            if 'predictor' not in name:   
                param.requires_grad = False
            else:
                param.requires_grad=True

        model.predictor = nn.Sequential(nn.LayerNorm((768, ), eps=1e-12), 
                                         nn.Linear(768, 52))
        
        model.train()

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#         # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):

            running_loss = 0.0
            running_corrects = 0

            # Print epoch
            print(f'Starting epoch {epoch+1}')

            # Set current loss value
            current_loss = 0.0

            #progress_bar = tqdm(range(len(trainloader)))

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
            
            if (epoch+1) % 10 == 0:
                # Saving the model
                
                                # Process is complete.
                print('Saving trained model after epoch{}'.format(epoch+1))
                
                save_path = '/home/ubuntu/BEATs/beats/results/esc50-{}/finetuned-FULL_{}_epoch{}.pth'.format(data_type,data_type, epoch+1)
                torch.save(model.state_dict(), save_path)
                

                # Print about testing
                print('Starting testing')



            #def testing(self, dataset, batch_size, **kwargs):

                # Define data loaders for training and testing data in this fold
                testloader = torch.utils.data.DataLoader(
                                  testset, 
                                  batch_size=batch_size)

                # Load pre-trained model

        #         checkpoint = torch.load(f'/home/ubuntu/BEATs/beats/finetuned_models/beats/realworld/model-fold-4.pth')

        #         pdb.set_trace()
        #         cfg = BEATsConfig(checkpoint['cfg'])
        #         model = BEATs(cfg)
        #         model.load_state_dict(checkpoint['model'])

                # Evaluationfor this fold
                correct, total = 0, 0

                #df = pd.DataFrame(columns=['GT', 'predicted', 'confidence'])

                gt = []
                predicted_categories = []
                confidence = []

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
                        values, predicted = torch.max(outputs.data, 1)

                        # save the predicted outputs & confidences
                        gt.extend(targets.cpu().numpy())
                        predicted_categories.extend(predicted.cpu().numpy())
                        confidence.extend(values.cpu().numpy())

                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

                    df = pd.DataFrame({'GT': gt, 'predicted': predicted_categories, 'confidence': confidence})
                    df.to_csv("/home/ubuntu/BEATs/beats/results/esc50-{}/test_results_{}_epoch{}.csv".format(data_type, data_type, epoch+1), index=False)

                    # Print accuracy
                    print('Test Accuracy: %d %%' % (100.0 * correct / total))

    def __call__(self, input_dir, labels, data_type, holdout=None, **kwargs):
        
        if labels is None:
            labels = "/home/ubuntu/data/finetuning/labels/finetuning_{}_train_esc50_categories.csv".format(data_type)
        if holdout is None:
            holdout = "/home/ubuntu/data/finetuning/labels/finetuning_holdout_test_esc50_categories.csv"
          
        labels = pd.read_csv(labels)
        labels = labels.sample(frac=1).reset_index(drop=True)
        
        holdout = pd.read_csv(holdout)
        holdout = holdout.sample(frac=1).reset_index(drop=True)
            
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
        
        #self.training(dataset, **kwargs)
        
        
        ####Test data#####
        
        # Preprocess data
        X = list(holdout["filename"])
        y = list(holdout["category"])

        features = []

        for audio in self.preprocess_audio(audio_dir, X, **kwargs):
            features.extend(torch.Tensor(audio))
            torch.cuda.empty_cache()
        
        testset = Dataset(features, y)
        
        self.training(dataset, testset, data_type, **kwargs)
        
        #self.testing(dataset, **kwargs)
        
if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=1 << 6, help='batch size')
    parser.add_argument('-f','--input_dir')
    parser.add_argument('-l','--labels')

    parser.add_argument('-e', '--num_epochs', type=int, default=30)
    
    parser.add_argument('-m', '--model', default = 'models/fine-tuned-cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
    help='pre-trained model')
    parser.add_argument('-dt','--data_type')
    
    args = parser.parse_args()

    self = BEATsFineTuning()
    self(**vars(args))