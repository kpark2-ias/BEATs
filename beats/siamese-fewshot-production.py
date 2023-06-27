
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from transformers import AutoFeatureExtractor, ASTForAudioClassification, Trainer, AutoModelForAudioClassification

from BEATs import BEATs, BEATsConfig
from layers import SiameseNetwork, SiameseNetworkConcat

import pdb
import librosa
import numpy as np
import pandas as pd

from collections import defaultdict
from tqdm import tqdm
#from BEATs import BEATs, BEATsConfig

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        
        encoding = torch.tensor(self.encodings[idx])
        label = self.labels[idx]
        label = nn.functional.one_hot(torch.tensor(label, dtype=torch.long), num_classes=14)
        label = label.type(torch.FloatTensor)
        item = {'encoding': encoding, 'category': label}
        return item

    def __len__(self):
        return len(self.encodings)

def preprocess_input(shot):
    
    ####target data##########
    train = pd.read_csv("/home/ubuntu/data/audio_event_06202023/audio_event_06202023_chunked_train.csv")
    #test = pd.read_csv("/home/ubuntu/data/audio_event_06202023/audio_event_06202023_chunked_test.csv")[:10]
    test = pd.read_csv("/home/ubuntu/data/audioset/audioset_fewshot_test_prod.csv")
    group = train.groupby('category')
    train = pd.DataFrame(group.head(shot)).groupby('category') 
    #test = pd.DataFrame(group.tail(1)).groupby('category') 
    #test = test.groupby('category')

    #########For esc-50#############
#     data = pd.read_csv("~/data/ESC-50-master/meta/esc50.csv")

#     group = data.groupby('category')
#     train = pd.DataFrame(group.head(shot))
#     train = train.set_index('filename')
#     test = data[~data['filename'].isin(train.index)]
#     train = train.reset_index()
#     train = train.groupby('category')
#     test = test.groupby('category')

    
    
    print("test_len : ", len(test))
    return train, test

def compute_metrics(df, category, model_name, shot, th):
    

    import sklearn.metrics
    df_concated = pd.DataFrame({'category': category})
    precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(df["GT"], df["predicted"], labels=category)
    acc = sklearn.metrics.accuracy_score(df["GT"], df["predicted"])
    #len(df[df["GT"] == df["predicted"]])/len(df)
    
    df2 = pd.DataFrame({'model':model_name, 'shot':shot, 'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'support':support})
    
    df_concated = pd.concat([df_concated, df2], axis=1)
    df_concated.to_csv(f"results/siamese_fewshot_results/test-prod_th_{th}/{model_name}/{mode}/result_shot_{shot}_argmax.csv", index=False)
    
    import sklearn.metrics
    df_concated = pd.DataFrame({'category': category})
    precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(df["GT"], df["predicted_multi"], labels=category)
    acc = sklearn.metrics.accuracy_score(df["GT"], df["predicted_multi"])
    #len(df[df["GT"] == df["predicted"]])/len(df)
    
    df2 = pd.DataFrame({'model':model_name, 'shot':shot, 'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'support':support})
    
    df_concated = pd.concat([df_concated, df2], axis=1)
    df_concated.to_csv(f"results/siamese_fewshot_results/test-prod_th_{th}/{model_name}/{mode}/result_shot_{shot}_threshold.csv", index=False)

def main(model_name, labels, input_dir, shot, mode, threshold):
    
    data = pd.read_csv(labels)
    cat = ['Ambulance (siren)','Crying, sobbing','Explosion','Gunshot, gunfire','Laughter','Screaming', 'Smash, crash']
    data = data.sample(frac=1).reset_index(drop=True)

    train, test = preprocess_input(shot)
    
    if model_name == 'BEATs':
        model_filename = 'fine-tuned-cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
        checkpoint = torch.load(f'models/{model_filename}')
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
    mu = {}
    q = defaultdict(lambda: [])

    ##support set
    x_j = []
    y_j = []

    with torch.no_grad():
        for index, category in enumerate(tqdm(train)):
            outputs = []
            for filepath in category[1]['filepath']:
#                 if category[0] in cat:#source == 'audioset':
#                     track, _ = librosa.load(f'{input_dir}/AudioSet/train_wav/{filename}.wav', sr=16000, dtype=np.float32)
#                 else:
#                     track, _ = librosa.load(f'{input_dir}/{category[0]}/_chunked/{filename}', sr=16000, dtype=np.float32)
                
                track, _ = librosa.load(f'{filepath}', sr=16000, dtype=np.float32)

                input_tensor = torch.from_numpy(track)

                if model_name == 'BEATs':
                    input_tensor = input_tensor.reshape(1, input_tensor.shape[0])
                    padding_mask = torch.zeros(len(input_tensor), input_tensor.shape[1]).bool().to('cuda')
                    output = model.extract_features(input_tensor, padding_mask)[0]
                elif 'AST' in model_name:
                    f = feature_extractor(input_tensor, sampling_rate=16000, return_tensors="pt")
                    output = model(**f).logits
                   
                x_j.append(output)
                y_j.append(index)
                outputs.append(output)
                torch.cuda.empty_cache()
            outputs = torch.stack(outputs)
            outputs = torch.reshape(outputs, (outputs.shape[0], -1))
            outputs = torch.mean(outputs, 0)
            #outputs = nn.functional.normalize(outputs, dim=0)
            mu[category[0]] = outputs

        trainset = Dataset(torch.stack(x_j), y_j)
        
#         for category in tqdm(test):
#             for filename in category[1]['filename']:
# #                 if category[0] in cat:
# #                     track, _ = librosa.load(f'{input_dir}/AudioSet/train_wav/{filename}.wav', sr=16000, dtype=np.float32)
# #                 else:
# #                     track, _ = librosa.load(f'{input_dir}/{category[0]}/_chunked/{filename}', sr=16000, dtype=np.float32)
                
#                 track, _ = librosa.load(f'{input_dir}/{filename}', sr=16000, dtype=np.float32)
        
#                 input_tensor = torch.from_numpy(track)
#                 if model_name == 'BEATs':
#                     input_tensor = input_tensor.reshape(1, input_tensor.shape[0])
#                     padding_mask = torch.zeros(len(input_tensor), input_tensor.shape[1]).bool().to('cuda')
#                     output = model.extract_features(input_tensor, padding_mask)[0]
#                 elif 'AST' in model_name:
#                     f = feature_extractor(input_tensor, sampling_rate=16000, return_tensors="pt")
#                     output = model(**f).logits
#                 output = torch.flatten(output)
#                 #output = nn.functional.normalize(output, dim=0)
#                 torch.cuda.empty_cache()
#                 q[category[0]].append(output)
    index = {}
    cols = [] 
    
    for idx, cat in enumerate(mu):
        index[idx] = cat
        cols.append(cat)

    ####Test#####
    ground_truths = []
    predicted_max = []
    predicted_multilabel = []
    predicted_prob = []
    predicted_binary = []
    
    if mode == 'diff':
        model_sim = SiameseNetwork()
    elif mode == 'concat':
        model_sim = SiameseNetworkConcat()
    #threshold = 0.99
#     if model_name == 'BEATs':
#         threshold = 0.553184
#     elif model_name == 'AST':
#         threshold = 0.571855
#     elif model_name == 'Distil-AST':
#         threshold = 0.651773
        
    model_sim.load_state_dict(torch.load(f'./siamese_models/{model_name}/{mode}/model-fold-0.pth'))
    model_sim.eval()
    
    
    for idx, row in tqdm(test.iterrows()):

        filepath = row['filepath']
        gt = row['category']
        track, _ = librosa.load(f'{filepath}', sr=16000, dtype=np.float32)

        input_tensor = torch.from_numpy(track)
        if model_name == 'BEATs':
            input_tensor = input_tensor.reshape(1, input_tensor.shape[0])
            padding_mask = torch.zeros(len(input_tensor), input_tensor.shape[1]).bool().to('cuda')
            output = model.extract_features(input_tensor, padding_mask)[0]
        elif 'AST' in model_name:
            f = feature_extractor(input_tensor, sampling_rate=16000, return_tensors="pt")
            output = model(**f).logits
        output = torch.flatten(output)
        #output = nn.functional.normalize(output, dim=0)
        
        outputs = []
        for cat_m in mu:
            outputs.append(torch.flatten(model_sim(mu[cat_m].reshape((1, -1)),output.reshape((1,-1)))))
        
        outputs = torch.stack(outputs)
        predicted_prob.append(torch.flatten(outputs).tolist())
        binary = torch.where(outputs > threshold, 1, 0)
        binary = torch.flatten(binary).tolist()
        predicted_binary.append(binary)
        torch.cuda.empty_cache()
        ###multi-label###

        result_max_sim = torch.argmax(outputs).item()

        ground_truths.append(gt)
        predicted_max.append(index[result_max_sim])
        torch.cuda.empty_cache()

        pred = ''

        for i, binary_output in enumerate(binary):
            if binary_output == 1:
                if pred == '':
                    pred += index[i]
                else: 
                    pred += "\t" + index[i]

        if pred == '':
            pred = 'Negative'

        predicted_multilabel.append(pred)
        

    df = pd.DataFrame()
    df_binary = pd.DataFrame(predicted_binary, columns = cols)
    df_prob = pd.DataFrame(predicted_prob, columns = cols)
    df["GT"] = ground_truths
    df["predicted"] = predicted_max
    df["predicted_multi"] = predicted_multilabel
    
    result = pd.concat([df, df_binary, df_prob], axis=1)
    result.to_csv(f"results/siamese_fewshot_results/test-prod_th_{threshold}/{model_name}/{mode}/result_shot_{shot}_multilabel_output.csv", index=False)
    
    category = cols
    compute_metrics(df, category, model_name, shot, threshold)

    acc = len(df[df["GT"] == df["predicted"]])/len(df)
    
    print(f"------------------test before training: {acc}")
    
if __name__ == "__main__": 
    
    from argparse import ArgumentParser
    import torch
    from torch import nn
    parser = ArgumentParser()
    parser.add_argument('-i','--input_dir', default='/home/ubuntu/data/audio_event_06202023/wav/_chunked')
    parser.add_argument('-l','--labels', default='/home/ubuntu/data/audio_event_06202023/audio_event_06202023_chunked.csv')
    parser.add_argument('-n','--shot', default=5)
    parser.add_argument('-m', '--model_name', default = 'AST')
    args = parser.parse_args()
    
    for i in [1, 3, 5]:
        for th in [0.98, 0.97, 0.96, 0.99]: 
            for model_name in ["AST", "Distil-AST"]:
                dir_path = f"results/siamese_fewshot_results/test-prod_th_{th}"
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                
                sub_path = f"results/siamese_fewshot_results/test-prod_th_{th}/{model_name}"
                if not os.path.exists(sub_path):
                    os.mkdir(sub_path)
                
                for mode in ["diff"]:
                    sub_dir = sub_path+f"/{mode}"
                    if not os.path.exists(sub_dir):
                        os.mkdir(sub_dir)

                    args.shot = i
                    args.threshold = th
                    args.model_name = model_name
                    args.mode = mode
                    main(**vars(args))