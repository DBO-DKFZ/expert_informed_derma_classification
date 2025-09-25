from pathlib import Path
import numpy as np
import schedulefree
from torchvision import transforms
from DermaClassifier.utils.ImbalancedDatasetSampler import ImbalancedDatasetSampler
import DermaClassifier.utils.hyperparmeter as hp


### CHANGE FOR HYPERPARAMETER ###############################################################

preprocess = "rgb_darker" # normal, rgb_darker, rgb_contrast, rgb_gray
encode_label = "ohe" # ohe => One-Hot-Encoding | sl => soft-label

# Define these variables with the information about where you added the pathological panel information.
patho_panel_tabel = "Labels_Panel-assured_Ground_Truth_Paper.xlsx"  # Name of the pathological panel.
data_path = "../../scp_data/"  # Path to data and table information.
split_path = "./data_split/"  # Path to the train, test and val set with lesoin ids.

############################################################################################

local_path = "sqlite:////" + str(Path(__file__).resolve().parent.parent.parent / "optuna.db")

mask = np.load("src/DermaClassifier/utils/mask.npy", allow_pickle=True)

gpu = 0  # Which Gpu device to use
gpu_num = 0 # How many gpus to use.

worst_val_prediction_list_len = 10
tensorboard_show_img = False


# Image prepare for model interpretation
in_imgs_size = 600  # Image size to load, can be choosen between 500 and 600
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))
color_darker_factor = 0.4  # Factor for darker lesion
add_contrast_factor = 2  # Factor for higher contrast
approx = 1  # Treshhold for turning point calculation
all_data = True  # Define if all images of a lesion should be taken, or pick only one random image out of six


# Definition of multiclass problem
encoding = {'melanoma': 0, 'in-situ tumor': 1, 'nevus': 2}
num_classes = len(encoding)


# Early stopping parameter
patience = 15
min_delta = 0.01


# Hyperparameter
encode = hp.encode(encode_label)
diagnosis_label = hp.diagnosis_label(encode_label)
do_sampler = hp.do_sampler(preprocess, encode)
sampling = hp.do_sampler(preprocess, encode)
weight_loss = hp.weight_loss(preprocess, encode)
lr = hp.lr(preprocess, encode)
wd = hp.wd(preprocess, encode)
bs = hp.bs(preprocess, encode)


optimize =  lambda param, lr, wd : schedulefree.AdamWScheduleFree([{"params": param, 
                                                 "lr": lr}]) if wd is None else schedulefree.AdamWScheduleFree([{"params": param, 
                                                                                              "lr": lr}], weight_decay=wd)

sampler = lambda train_set, samp_factor, seed: ImbalancedDatasetSampler(dataset=train_set, sampling_factor=samp_factor, num_classes=num_classes, 
                                                                  labels=[encoding[i.class_name] for i in train_set.data], shuffle=True, 
                                                                  seed=seed)
