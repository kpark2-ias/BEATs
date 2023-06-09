import pandas as pd
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from pytorchtools import EarlyStopping
from transformers import AutoFeatureExtractor, ASTForAudioClassification, Trainer, AutoModelForAudioClassification

from BEATs import BEATs, BEATsConfig
import pdb
import librosa
import numpy as np
from layers import SiameseNetwork, SiameseNetworkConcat

from collections import defaultdict
from tqdm import tqdm

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings_1, encodings_2, labels=None):
        self.encodings_1 = encodings_1
        self.encodings_2 = encodings_2
        self.labels = labels

    def __getitem__(self, idx):
        
        encoding_1= torch.tensor(self.encodings_1[idx])
        encoding_2= torch.tensor(self.encodings_2[idx])
        label = self.labels[idx]
        item = {'input1': encoding_1, 'input2': encoding_2, 'label': label}
        return item

    def __len__(self):
        return len(self.encodings_1)

def preprocess_input(data, model_name, input_dir):
    
    
        ##support set
        
    if model_name == 'BEATs':
        model_filename = 'fine-tuned-cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
        checkpoint = torch.load(f'/home/ubuntu/BEATs/beats/models/{model_filename}')
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
    elif model_name == 'AST':
        feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    elif model_name == 'Distil-AST':
        feature_extractor = AutoFeatureExtractor.from_pretrained("bookbot/distil-ast-audioset")
        model = AutoModelForAudioClassification.from_pretrained("bookbot/distil-ast-audioset")

    model.eval()
    
    encodings1 = []
    encodings2 = []
    y = []

    with torch.no_grad():
        for index, row in tqdm(data.iterrows()):
            input1 = row['f1']
            input2 = row['f2']
        
            track1, _ = librosa.load(f'{input_dir}/AudioSet/train_wav/{input1}.wav', sr=16000, dtype=np.float32)
            track2, _ = librosa.load(f'{input_dir}/AudioSet/train_wav/{input2}.wav', sr=16000, dtype=np.float32)
            
            input_tensor1 = torch.from_numpy(track1)
            input_tensor2 = torch.from_numpy(track2)

            if model_name == 'BEATs':
                input_tensor1 = input_tensor1.reshape(1, input_tensor1.shape[0])
                padding_mask1 = torch.zeros(len(input_tensor1), input_tensor1.shape[1]).bool().to('cuda')
                output1 = model.extract_features(input_tensor1, padding_mask1)[0]

                input_tensor2 = input_tensor2.reshape(1, input_tensor2.shape[0])
                padding_mask2 = torch.zeros(len(input_tensor2), input_tensor2.shape[1]).bool().to('cuda')
                output2 = model.extract_features(input_tensor2, padding_mask2)[0]

            elif 'AST' in model_name:
                f1 = feature_extractor(input_tensor1, sampling_rate=16000, return_tensors="pt")
                output1 = model(**f1).logits

                f2 = feature_extractor(input_tensor2, sampling_rate=16000, return_tensors="pt")
                output2 = model(**f2).logits

            encodings1.append(output1)
            encodings2.append(output2)
            y.append(row['label'])
            torch.cuda.empty_cache()
                
        trainset = Dataset(torch.stack(encodings1), torch.stack(encodings2), y)
    
    return trainset


def training(dataset, num_epoch, batch_size, model_name, mode):
    k_folds = 5
    num_epochs = num_epoch
    loss_function = nn.BCELoss()#nn.CrossEntropyLoss()
    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        
        val_ids, test_ids = np.array_split(test_ids,2)
        
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                          dataset, 
                          batch_size=batch_size, sampler=train_subsampler)
        validationloader = torch.utils.data.DataLoader(
                          dataset, 
                          batch_size=batch_size, sampler=val_subsampler)
        testloader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=batch_size, sampler=test_subsampler)
        
        if mode == 'diff':
            model = SiameseNetwork()
        elif mode == 'concat':
            model = SiameseNetworkConcat()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        early_stopping = EarlyStopping(patience=5, verbose=True)
        
        for epoch in range(0, num_epochs):
            
            #### Train the model###
            model.train()
            
            running_loss = 0.0
            running_corrects = 0

            # Print epoch
            print(f'Starting epoch {epoch+1}')

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for data in trainloader:

                # Get inputs
                inputs1 = data['input1']
                inputs2 = data['input2']
                targets = data['label']

                inputs1 = torch.tensor(inputs1)
                inputs2 = torch.tensor(inputs2)
                targets = torch.tensor(targets).type(torch.FloatTensor).to('cuda')
                optimizer.zero_grad()
                outputs = model(inputs1, inputs2)
                
                outputs = torch.flatten(outputs)
                
                preds = torch.where(outputs > 0.5, 1, 0)
                
                loss = loss_function(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()*inputs2.size(0)
                running_corrects += torch.sum(preds==targets)

            epoch_loss = running_loss / len(trainloader.dataset)
            epoch_acc = running_corrects.double() / len(trainloader.dataset)
            print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc)) 
            
            #### Validate the model###
            model.eval()
            
            running_val_loss = 0.0
            running_val_corrects = 0
            
            for data in validationloader:

                # Get inputs
                inputs1 = data['input1']
                inputs2 = data['input2']
                targets = data['label']

                inputs1 = torch.tensor(inputs1)
                inputs2 = torch.tensor(inputs2)

                targets = torch.tensor(targets).type(torch.FloatTensor).to('cuda')

                outputs = model(inputs1, inputs2)
                outputs = torch.flatten(outputs)
                preds = torch.where(outputs > 0.5, 1, 0)

                loss = loss_function(outputs, targets)

                running_val_loss += loss.item()*inputs2.size(0)
                running_val_corrects += torch.sum(preds==targets)

            val_loss = running_val_loss / len(validationloader.dataset)
            val_acc = running_val_corrects.double() / len(validationloader.dataset)
            print('----Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
            
            early_stopping(val_loss, model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        # Saving the model
        save_path = f'./siamese_models/{model_name}/{mode}/model-fold-{fold}.pth'
        torch.save(model.state_dict(), save_path)
        
        correct, total = 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):

                # Get inputs
                inputs1 = data['input1']
                inputs2 = data['input2']
                targets = data['label']

                inputs1 = torch.tensor(inputs1)
                inputs2 = torch.tensor(inputs2)

                targets = torch.tensor(targets).type(torch.FloatTensor).to('cuda')

                model.eval()
                outputs = model(inputs1, inputs2)
                outputs = torch.flatten(outputs)
                preds = torch.where(outputs > 0.5, 1, 0)

                total += targets.size(0)
                correct += (preds == targets).sum().item()
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
    with open(f'./siamese_models/{model_name}/{mode}/result.txt', 'w') as f:
        f.write(f'5CV Average Accuracy: {sum/len(results.items())} %')

def main(model_name, labels, input_dir, num_epoch, batch_size, mode):
    
    data = pd.read_csv(labels)
    data = data.sample(frac=1).reset_index(drop=True)
    
    for model_name in ["BEATs", "AST","Distil-AST"]:
        if not os.path.exists(f"./siamese_models/{model_name}"):
            os.mkdir(f"./siamese_models/{model_name}")
        trainset = preprocess_input(data, model_name, input_dir)
        for mode in ["concat", "diff"]:
            if not os.path.exists(f"./siamese_models/{model_name}/{mode}"):
                os.mkdir(f"./siamese_models/{model_name}/{mode}")
            training(trainset, num_epoch, batch_size, model_name, mode)

# model_name = 'AST'
# #labels = '/home/ubuntu/data/_ipynb/siamese_train_200k.csv'
# labels = '/home/ubuntu/data/_ipynb/siamese_train.csv'
# input_dir = '/home/ubuntu/data/wav'
# batch_size = 32
# num_epoch = 100

# main(model_name, labels, input_dir, num_epoch, batch_size)

if __name__ == "__main__": 
    
    from argparse import ArgumentParser
    import torch
    from torch import nn
    import os 
    
    parser = ArgumentParser()
    parser.add_argument('-i','--input_dir', default='/home/ubuntu/data/wav')
    parser.add_argument('-l','--labels', default='/home/ubuntu/data/_ipynb/siamese_train_200k.csv')
    #parser.add_argument('-n','--shot', default=5)
    parser.add_argument('-b', '--batch_size', default = 32)
    #parser.add_argument('-f','--finetuning', default=None)
    parser.add_argument('-e', '--num_epoch', type=int, default=100)
    parser.add_argument('-m', '--model_name', default = 'Distil-AST')
    parser.add_argument('-md', '--mode', default='diff')
    args = parser.parse_args()

    
    main(**vars(args))

