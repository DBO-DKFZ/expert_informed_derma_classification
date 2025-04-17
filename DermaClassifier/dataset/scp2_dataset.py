"""
Dataset creation for the pathological panel.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import argparse

from DermaClassifier.dataset import dataset_helper
from DermaClassifier.utils import config

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SCP2Dataset(Dataset):
    def __init__(self, args: argparse.ArgumentParser, dataset_type: str, demo: bool=False):
        """
        Class create the dataset for training, testing or validation.
        
        :args: Namespace information how to setup the dataclass.
        :dataset_type: information if dataclass is for training, testing (holdout, external) or validation.
        :demo: True, if the data is from the little demonstation example.
        """
        self.args = args
        self.dataset_type = dataset_type  # define if "train", "val", or "test" case
        self.demo = demo

        if not self.demo:
            self.set_up_data_table()
            self.patientId = np.array(self.table["patientId"].tolist())
            self.slideId = np.array(self.table["slideId"].tolist())
        else:
            self.set_up_demo_table()

        self.preprocess_image = self.get_preprocess()
        self.transform = self.get_transformation()
        
        self.not_found_ids = {}
        self.data = self.load_data(self.table)
    
    def set_up_data_table(self):
        """ Load table of the pathological panel and depending the dataset tpye the file with the used slideIds. """
        data_file = Path(config.split_path, self.dataset_type + ".npy")
        data_split = np.load(data_file, allow_pickle=True)

        self.table = pd.read_excel(self.args.table_path, header=[0, 1, 2])
        self.table.columns = self.table.columns.values
        self.table.rename(columns={(ele[0], ele[1], ele[2]): (ele[2] if idx < 3 else (ele[0], ele[2])) for idx, ele in enumerate(self.table.columns.values)},
                          inplace=True)
        self.table = self.table[self.table["slideId"].astype(str).isin(data_split.astype("str"))].reset_index(drop=True)
        self.table = self.table.rename(columns={('PP Majority Vote ', 'lesionId'): "lesionId"})
        self.table = self.table[self.table[('PP Majority Vote ', 'label majoriy vote')].apply(lambda row: row in list(config.encoding.keys()))]
    
    def set_up_demo_table(self):
        """ Load table for the demo. """
        self.table = pd.read_csv(Path("demo", "demo_label_data.csv"))
        self.table = self.table[self.table.type == self.dataset_type]

    def load_data(self, table: pd.DataFrame) -> list:
        """ Load the images with the label. """
        data = []

        for idx in tqdm(range(table.shape[0])):
            sample = table.iloc[idx]  # get the whole line of the dataset
            not_found = []

            if self.dataset_type != "train" and not config.all_data:
                data_sample = []
            
            # Go over the six images of the actual lesion
            for image_ids in sample["imageIds"].strip('][').split(', '):

                if not os.path.exists(os.path.join(self.args.images_path, image_ids[1:-1]+ ".png")):
                    not_found.append(image_ids)
                    continue
                
                # Define as datapoint with laben and image
                data_point = dataset_helper.DataSample(name=image_ids[1:-1],
                                                        info=sample, 
                                                        label_mode="majority" if "test" in self.dataset_type else self.args.diagnosis_label,
                                                        img_path=self.args.images_path,
                                                        demo=self.demo)
                
                if self.dataset_type == "train" or config.all_data:
                    data.append(data_point)
                else:
                    data_sample.append(data_point)
            
            if self.dataset_type != "train" and not config.all_data:
                data.append(data_sample)

            if not_found != []:
                self.not_found_ids[sample["lesionId"]] = not_found

        return data

    def  __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        # load imag and label
        if self.dataset_type == "train" or config.all_data:
            image = self.data[index].image
            label = self.data[index].label  
        else:
            image = self.data[index][np.random.randint(6)].image
            label = self.data[index][0].label
    
        # change color 
        image = self.preprocess_image(image)
        
        image = Image.fromarray(image)
        image = self.transform(image)
        
        return image, label
    
    def get_item_pred(self, index: int):
        """ Get an item of the dataset for testcases. """
        if torch.is_tensor(index):
            index = index.tolist()
        
        image = self.data[index].image
        label = self.data[index].label
    
        # change color 
        image = self.preprocess_image(image)
        
        image = Image.fromarray(image)
        image = self.transform(image)
        
        return image, label, self.data[index]

    def get_preprocess(self):
        """ Depending if or if not a preprocessing is done. """
        if config.preprocess == "rgb_gray":
            return lambda x: dataset_helper.gray_image_region_darker(x)
        elif config.preprocess == "rgb_darker":
            return lambda x: dataset_helper.color_image_region_darker(x)
        elif config.preprocess == "rgb_contrast":
            return lambda x: dataset_helper.add_contrast_to_image(x)
        else:
            return lambda x : x
    
    def get_transformation(self):
        """ Set the transformations for image augmentation. """
        transform = []

        if config.preprocess == "rgb_gray":
            transform.append(transforms.Grayscale(num_output_channels=3))

        if self.dataset_type == "train":
            transform += [transforms.Resize((config.in_imgs_size, config.in_imgs_size)),
                          transforms.ToTensor(),
                          transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomVerticalFlip(p=0.5),
                          transforms.RandomRotation(90),
                          transforms.Normalize(config.mean, config.std),
                         ] 
        else:    
            transform += [transforms.Resize((config.in_imgs_size, config.in_imgs_size)),
                          transforms.ToTensor(),
                          transforms.Normalize(config.mean, config.std),
                          ]

        return transforms.Compose(transform)
