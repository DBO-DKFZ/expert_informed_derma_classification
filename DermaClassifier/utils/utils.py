import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import re
import random
import argparse

import DermaClassifier.utils.config as config
import DermaClassifier.utils.hyperparmeter as hp


def fix_randomness(seed:int=42):
    """ Set the randomness to specific seed for reproducible results. """
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> str:
    """ Depending if GPU exists set device to GPU or CPU. """
    if torch.cuda.is_available():
            device = f"cuda:{config.gpu}"
    else:
        device = "cpu"
    print(f'Device defined as {device}')
    return device


def imshow(img: np.array):
  ''' function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


def moving_average(data: np.array, window_size: int) -> np.array:
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def get_saving_path(args: argparse.ArgumentParser):
    """ Define name of optimization path. """
    label_mode = args.diagnosis_label
    dir_model_name = re.compile("([a-zA-Z]+)([0-9]+)").match(args.model).group(1)
    implement_mode = "optimize"
    path = Path(args.save_path, label_mode, implement_mode)
    
    return path


def image_side_by_side(img_a: np.array, img_b: np.array):
    assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
    assert img_a.dtype == img_b.dtype
    h, w, c = img_a.shape
    canvas = np.zeros((h, 2 * w, c), dtype=img_a.dtype)
    canvas[:, 0 * w:1 * w, :] = img_a
    canvas[:, 1 * w:2 * w, :] = img_b
    return canvas


class EarlyStopper:
    def __init__(self, patience:int , verbose:bool=False, min_delta:int=0):
        """ Define Earlystopper for training process, if AUROC is not changing anymore. """
        self.patience = patience
        self.verbose = verbose
        
        self.counter = 0

        self.best_loss = None
        self.early_stop = False
        self.min_delta = min_delta
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif np.abs(self.best_loss - val_loss) <= self.min_delta: 
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def set_hyperparameter_demo(opt: argparse.ArgumentParser, 
                            preprocess: str, encode_label: str) -> argparse.ArgumentParser:
    """ Get our hyperparameter for demo in jupyter notebook. """

    encode = hp.encode(encode_label)
    
    # Hyperparameter
    opt.diagnosis_label = hp.diagnosis_label(encode_label)
    opt.lr = hp.lr(preprocess, encode)
    opt.wd = hp.wd(preprocess, encode)
    opt.batch_size = hp.bs(preprocess, encode)
    
    opt.sampling = 0
    opt.do_sampler = False
    opt.continue_train = False
    opt.optimize = False
    
    return opt
