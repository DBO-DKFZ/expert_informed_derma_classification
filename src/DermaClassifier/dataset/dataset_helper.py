import numpy as np
from PIL import Image
import os
import cv2
import copy
import pandas as pd
from DermaClassifier.utils import utils, config


class DataSample:
    """ Definition of one datapoint that is feed into the model and is read from the pathological panel table. """
    
    def __init__(self, name: str, info: pd.Series , label_mode: str, img_path: str, demo:bool=False):
        """
        Class to load image with the specific label encoded as majority or soft label.
        
        :name: image name of thes lesion.
        :info: includes all information of the lesion with the diagnosies of the pathological panel.
        :label_mode: info how to encode the label.
        :img_path: path of the image.
        :demo: True, if the data is from the little demonstation example.
        """
        self.name = name
        self.info = info
        self.demo = demo
        
        self.label_mode = label_mode
        self.label = self.encode(label_mode)
        
        self.img_path = os.path.join(img_path, self.name + ".png")

    @property
    def image(self) -> np.array:
        """ Load the dermatoscopic image of the SCP2 dataset and transform into a numpy array. """
        return np.array(Image.open(self.img_path).convert("RGB"))

    def encode(self, label_mode: str) -> np.array:
        """ Encode the label as one-hot encoding or soft-label version. """

        # Do the encoding by distributional information to create a soft-label
        if label_mode == "dist":
            return self.dist_encoding()
        # Create the one-hot encoding of the class with the majority/reference of the pathological panel
        elif label_mode == "majority":
            return self.majority_encoding()
        else:
            raise NotImplementedError

    def one_hot_encoding(self, l: str) -> np.array:
        """ Create a one-hot-encoding label and set the dimension to hot (1) dependant on the diagnsois. """
        label = np.zeros(config.num_classes)
        label[config.encoding[l]] = 1
        return label
    
    def dist_encoding(self):
        """ Encode the label as soft label to include the uncertainty of the pathological panel. """
        # Get all the diagnosis of the pathological panel
        labels = [self.info[ele] for ele in self.info.keys() if "softlabel" in ele] if self.demo else [self.info[ele] for ele in self.info.keys() if "Reviewer" in ele[0] and ele[1] == "diagnosis"] 
        # Set class name with the majority vote of the pathological panel
        self.class_name = self.info["diagnosis"] if self.demo else self.info[('PP Majority Vote ', 'label majoriy vote')]
        # Get a list of all the diagnosis
        labels = [item for item in labels if item in config.encoding]
        self.multi_class_name = list(set(labels))
        # Create the soft-label
        return np.sum(np.array([self.one_hot_encoding(l) for l in labels]), axis=0) / 8
    
    def majority_encoding(self):
        """ Encode the label as one-hot-encoding dependig on the majority vote of the pathological panel. """
        # Get the majority vote of the pathological panel from our table
        label = self.info["diagnosis"] if self.demo else self.info[('PP Majority Vote ', 'label majoriy vote')]
        self.class_name = label
        return self.one_hot_encoding(label) 


def get_color_turning_point_idx(gray_image: np.array) -> int:
    """ Compute the turning point of a histograph of colors """
    # Compute the histogram of the image
    hist_color = cv2.calcHist([gray_image], [0], config.mask, [256], [0, 256]).reshape(-1)
    # Create the function f of the histogram
    y_smoothed = utils.moving_average(hist_color, 30)
    x = np.arange(len(y_smoothed))

    # Compute the first and second derivative of the function
    dy = utils.moving_average(np.gradient(y_smoothed, x), 30)
    d2y = utils.moving_average(np.gradient(dy, x), 30)

    # Determine an interesting interval to prohibit irrelevant area is included from the border
    lower_val_border, higher_val_border = np.where(np.round(d2y)!=0)[0][0], np.where(np.round(d2y)!=0)[0][-1]
    turning_points = np.where(np.abs(np.round(d2y, 3))[lower_val_border:higher_val_border] < config.approx)[0]
    idx = int(turning_points[int(len(turning_points)/2)] + lower_val_border)

    return idx


def gray_image_region_darker(image: np.array) -> np.array:
    """ Get gray image and color lesion region darker by determining color change point. """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    idx = get_color_turning_point_idx(gray_image)

    # Color every pixel darker that is smaller than the turning point
    gray_image = gray_image.astype("float")
    gray_image[gray_image < idx] *= config.color_darker_factor
    
    return gray_image


def color_image_region_darker(image: np.array) -> np.array:
    """ Color lesion region darker by determining color change point in RGB image. """
    gray_image = cv2.cvtColor(copy.deepcopy(image), cv2.COLOR_RGB2GRAY)
    idx = get_color_turning_point_idx(gray_image)

    # Color every pixel darker that is smaller than the turning point
    color_mask = gray_image < idx
    image[color_mask] = image[color_mask] * config.color_darker_factor
    
    return image


def add_contrast_to_image(image: np.array) -> np.array:
    """ Increase contrast in RGB image. """
    gray_image = cv2.cvtColor(copy.deepcopy(image), cv2.COLOR_RGB2GRAY)
    idx = get_color_turning_point_idx(gray_image)

    # Get the pixel position which we want to set the contrast higher
    contrast = np.logical_and(gray_image < idx, np.any(image != np.array([0, 0, 0]), axis=2))
    contrast_count = contrast.sum()
    
    contrast_img = copy.deepcopy(image)
    contrast_img[~contrast] = 0

    # Compute the R, G, B values to set contrast higher 
    mean_r, mean_g, mean_b = np.sum(contrast_img[:, :, 0])/contrast_count, np.sum(contrast_img[:, :, 1])/contrast_count,np.sum(contrast_img[:, :, 2])/contrast_count

    size_x, size_y, _ = image.shape
    mean_r_mask = mean_g_mask = mean_b_mask = np.zeros([size_x, size_y])
    mean_r_mask[contrast] = mean_r
    mean_g_mask[contrast] = mean_g
    mean_b_mask[contrast] = mean_b

    R = np.maximum(0, np.minimum(255, ((contrast_img[:, :, 0] - mean_r_mask) * config.add_contrast_factor + mean_r_mask).astype("uint8")))
    G = np.maximum(0, np.minimum(255, ((contrast_img[:, :, 1] - mean_g_mask) * config.add_contrast_factor + mean_g_mask).astype("uint8")))
    B = np.maximum(0, np.minimum(255, ((contrast_img[:, :, 2] - mean_b_mask) * config.add_contrast_factor + mean_b_mask).astype("uint8")))

    result_img = np.zeros_like(image)
    result_img[~contrast] = image[~contrast]
    result_img[contrast] = np.stack([R, G, B], axis=2)[contrast]
    
    return result_img
