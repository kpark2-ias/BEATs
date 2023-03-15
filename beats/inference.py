import glob
import os
import librosa

from BEATs import BEATs, BEATsConfig
from utils.extract_audio import extract_audio
from tqdm import tqdm

import numpy as np
import torch
torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class BEATsInference(object):

    def __init__(self, model_filename = 'fine-tuned-cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt', verbose=False):
        torch.set_grad_enabled(False)
        
        self.checkpoint = torch.load(f'models/{model_filename}')#, map_location='cuda')
        cfg = BEATsConfig(self.checkpoint['cfg'])
        self.beats = BEATs(cfg)
        
        self.beats.load_state_dict(self.checkpoint['model'])
        self.beats.eval()

        if verbose:
            parameters = sum([x.numel() for x in self.beats.parameters()])/(10**6)
            print(f'Parameter count: {parameters:.1f}M')
    
    def preprocess_audio(self, input_dir,  SAMPLE_RATE = 44100, verbose=False, batch_size = 1 << 6, **kwargs):

        paths_to_audio = glob.glob(f'{input_dir}/*.wav')
        tracks = list()
        for idx, path_to_audio in enumerate(tqdm(paths_to_audio)):
            
            track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)
            track = torch.from_numpy(track)

            tracks.append(track)
            
            if not idx % batch_size:

                if verbose:
                    print([track.shape for track in tracks])
                    
                maxtrack =  max([ta.shape[-1] for ta in tracks])

                padded = [torch.nn.functional.pad(torch.from_numpy(np.array(ta)),(0,maxtrack-ta.shape[-1])) for ta in tracks]
                if verbose:
                    print( [track.shape for track in padded])

                audio = torch.stack(padded)

                if verbose:
                    print(audio.shape)
                
                yield audio 
                    
                tracks = list()
        
    def score_inputs(self, audio):

        padding_mask = torch.zeros(len(audio), audio.shape[1]).bool()
        probs = self.beats.extract_features(audio, padding_mask=padding_mask)[0]

        for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
            top5_label = [self.checkpoint['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
            print(f'Top 5 predicted labels of the {i}th audio are {top5_label} with probability of {top5_label_prob}')

    def __call__(self, input_dir=None, verbose=False, **kwargs):
        audio_dir = extract_audio(input_dir, **kwargs)
        for audio in self.preprocess_audio(audio_dir, verbose=verbose, **kwargs):
            self.score_inputs(audio)

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=1 << 6, help='batch size')
    parser.add_argument('-f','--input_dir')
    parser.add_argument('-n','--num',type=int,help='Limit to first N input files')
    args = parser.parse_args()

    self = BEATsInference()
    self(**vars(args))
