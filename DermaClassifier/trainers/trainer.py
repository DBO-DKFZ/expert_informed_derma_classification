import torch
import numpy as np
import os
from PIL import Image
import re
import argparse

from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC
from pathlib import Path

from DermaClassifier.utils import config
from DermaClassifier.utils.utils import EarlyStopper


class Trainer:

    def __init__(self, args: argparse.ArgumentParser, device: str, model, optimizer, 
                 train_set: DataLoader, val_set: DataLoader, 
                 weight_loss: torch.Tensor, trial_idx:int=None) -> None:
        """
        Training process.

        :args: Information of paths and training hyperparameter.
        :device: Train on cpu or gpu.
        :model: Torch model structure.
        :optimizer: Torch optimizer.
        :train_set: Torch dataloader with the training set.
        :val_set: Torch dataloader with the validation set.
        :weight_loss: Vector which dimension (class) is how high weighted.
        :trial_idx: Optimized trial index.
        """
        
        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.train_set = train_set
        self.val_set = val_set
        self.trial_idx = trial_idx
        self.weight_loss = weight_loss

        # Metrics
        self.acc_train = MulticlassAccuracy(num_classes=config.num_classes)
        self.acc_val = MulticlassAccuracy(num_classes=config.num_classes)
        self.f1_train = MulticlassF1Score(num_classes=config.num_classes)
        self.f1_val = MulticlassF1Score(num_classes=config.num_classes)
        self.auroc_train = MulticlassAUROC(num_classes=config.num_classes)
        self.auroc_val = MulticlassAUROC(num_classes=config.num_classes)

        self.loss_func = {"ce": nn.CrossEntropyLoss(weight=self.weight_loss.to(device)),
                          "kl": nn.KLDivLoss(reduction="batchmean"),
                          "mse": nn.MSELoss(),
                          "l1": nn.L1Loss()}[self.args.loss]

        self.encoding = dict()
        for k in config.encoding:
            self.encoding[config.encoding[k]] = k
        
        self.correct_pred = {"correct": 0, "wrong": 0}
    
    def train(self):
        """ Train logic. """
        self.best_vloss = 1_000_000.
        self.best_epoch = -1

        # saving path
        self.saving = self.get_saving_path()

        # Load the weights if training should continue
        if self.args.continue_train:
            load_data = torch.load(Path(self.saving, "last.ckpt"))
            self.model.load_state_dict(load_data["model_state_dict"])

        # Create path where weights should be saved, if it not exists
        if not os.path.exists(self.saving):
            os.makedirs(self.saving)

        # Define tenserboard overview to show training state
        self.writer = SummaryWriter(self.saving)

        # early_stopping
        early_stopper = EarlyStopper(patience=config.patience, verbose=True, min_delta=config.min_delta)
 
        for epoch in range(load_data['epoch']+1 if self.args.continue_train else 0, self.args.epochs):
            self.epoch = epoch
            self.train_epoch()

            early_stopper(self.val_acc_stop)
            if early_stopper.early_stop:
                break

        # Save the last model
        self.save_last_model()

        self.writer.flush()
        self.writer.close()
    
    def train_epoch(self):
        """ Logic of a training epoch. """

        # set model on train modus and turn gradient tracking on
        self.model.train()
        self.optimizer.train()

        losses = []

        for _, data in enumerate(self.train_set):     
            # train model with batch and loss of batch
            running_loss = self.train_batch(data)
            # append batch loss to list
            losses.append(running_loss)

        metric = {"train/train_loss_epoch": np.mean(np.array(losses)),
                  "train/train_acc_epoch": self.acc_train.compute().item()}

        validation_data = self.validate()
        self.loss_val = validation_data["val/val_loss"]
        self.val_acc_stop = validation_data['val/val_acc']

        self.writer.add_scalar("Loss/train", np.mean(np.array(losses)), self.epoch)
        self.writer.add_scalar("Accuracy/train", self.acc_train.compute().item(), self.epoch)
        self.writer.add_scalar("F1/train", self.f1_train.compute().item(), self.epoch)
        self.writer.add_scalar("AUROC/train", self.auroc_train.compute().item(), self.epoch)
        self.writer.add_scalar("Loss/val", validation_data['val/val_loss'], self.epoch)
        self.writer.add_scalar("Accuracy/val", validation_data['val/val_acc'], self.epoch)
        self.writer.add_scalar("F1/val", self.f1_val.compute().item(), self.epoch)
        self.writer.add_scalar("AUROC/val", self.auroc_val.compute().item(), self.epoch)


        self.save_best_model(validation_data["val/val_loss"])

        print("EPOCH {}/{}: TRAIN loss > {} \t accuracy > {} \t AUROC > {} | VAL loss > {} \t accuracy > {} \t AUROC > {}".format(self.epoch+1,
                                                                                         self.args.epochs,
                                                                                         np.round(metric["train/train_loss_epoch"], 3),
                                                                                         np.round(metric["train/train_acc_epoch"], 3),
                                                                                         np.round(self.auroc_train.compute().item(), 3), 
                                                                                         np.round(validation_data["val/val_loss"], 3), 
                                                                                         np.round(validation_data["val/val_acc"], 3),
                                                                                         np.round(self.auroc_val.compute().item(), 3),))

    def train_batch(self, data_pack: torch.Tensor) -> np.array:
        """
        data_pack: list
            Position 0 > represents the images of the batch
            Position 1 > represents the label of hthe images
        """
        img = data_pack[0].to(self.device)
        label = data_pack[1].to(self.device)

        # Transform image through the model
        output = self.model(img)
        loss = self.loss_func(output, label)

        # Get the predicted label and the label
        pred = torch.argmax(output, dim=1)
        target = torch.argmax(label, dim=1)
        self.acc_train.update(pred, target)
        self.f1_train.update(pred, target)
        self.auroc_train.update(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return np.abs(loss.item())
    

    def validate_batch(self, data_pack, batch_idx=0, show_img=False):
        """
        data_pack: list
            Position 0 > represents the images of the batch
            Position 1 > represents the label of hthe images
        batch_idx: int
            Get index to print specific images with predicted label in wandb
        show_img: boolean
            True if the images should be shown in wandb
        """

        # Check that model is not im training moduse
        assert self.model.training is False

        # Process in no training mode
        with torch.no_grad():
            image = data_pack[0].to(self.device)
            label = data_pack[1].to(self.device)
            
            # Transform through model
            output = self.model(image)
            
            # Compute loss
            vloss = self.loss_func(output, label)
            self.val_loss_iter.append(vloss.item())

            # Compute accuracy
            pred = torch.argmax(output, dim=1)
            target = torch.argmax(label, dim=1)
            self.acc_val.update(pred, target)
            self.f1_val.update(pred, target)
            self.auroc_val.update(output, target)
            
            if config.tensorboard_show_img:
                # check if loss is worse than last element of worse list
                if (np.abs(vloss.item()) > self.worst[-1] and pred != target).item():
                    worst_list = self.worst
                    # Go through list to find correct position
                    for idx, ele in enumerate(worst_list):
                        if np.abs(vloss.item()) > ele:
                            self.worst.insert(idx, np.abs(vloss.item()))
                            self.worst = self.worst[:config.worst_val_prediction_list_len]

                            show_img = (config.denormalize(image[0]).permute(1, 2, 0).cpu().numpy()*255).astype("int64")
                            pil_image = np.array(Image.fromarray((config.denormalize(image[0]).permute(1, 2, 0).cpu().numpy()*255).astype("uint8")).resize((250, 250)))
                            pred_label = self.encoding[torch.max(output.data, 1)[1].item()]
                            target_label = self.encoding[torch.max(label, 1)[1].item()]
                            self.worst_info.insert(idx, (show_img, pred_label, target_label, vloss.item()))
                            break
            
            if pred == target:
                self.correct_pred["correct"] += 1
            else:
                self.correct_pred["wrong"] += 1
    
    def validate(self):
        """ Function for validation of the val set. Call of the validation batch. """
        training = self.model.training
        
        # Change to evaluation mode
        self.model.eval()
        self.optimizer.eval()

        self.val_loss_iter = []

        # Lists to save images of validation set to show on wandb
        self.worst = [0]
        self.worst_info = []
        
        for batch_idx, data_val in enumerate(self.val_set):
            self.validate_batch(data_val, batch_idx)       

        validation_data = {'val/val_loss': np.mean(np.array(self.val_loss_iter)),
                           'val/val_acc': self.acc_val.compute().item()}

        self.correct_pred["correct"] = 0
        self.correct_pred["wrong"] = 0

        if training:
            self.model.train()
            self.optimizer.train()

        return validation_data

    def save_last_model(self):
        model_path = os.path.join(self.saving, f"last.ckpt")
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.args,
            'best_loss': self.best_vloss,
            'best_epoch': self.best_epoch,
            'image_mean': config.mean,
            'image_std': config.std,
            'encoding': config.encoding,
            'num_classes': config.num_classes,
            'weight_loss': self.weight_loss,
            'img_size': config.in_imgs_size,
            'preprocess': config.preprocess,
            'coloring_factor': config.add_contrast_factor if config.preprocess == "rgb_contrast" else config.color_darker_factor,
            'structure': self.args.model,
            'loss_func': str(self.loss_func)
        }, model_path)
    
    def save_best_model(self, loss):
        if loss < self.best_vloss:
            model_path = os.path.join(self.saving, f"best_epoch_{self.epoch}.ckpt")
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'args': self.args,
                'image_mean': config.mean,
                'image_std': config.std,
                'encoding': config.encoding,
                'num_classes': config.num_classes,
                'weight_loss': self.weight_loss,
                'img_size': config.in_imgs_size,
                'preprocess': config.preprocess,
                'coloring_factor': config.add_contrast_factor if config.preprocess == "rgb_contrast" else config.color_darker_factor,
                'structure': self.args.model,
                'loss_func': str(self.loss_func)
            }, model_path)

            # if exists delete an old best ckpt
            if os.path.exists(os.path.join(self.saving, f"best_epoch_{self.best_epoch}.ckpt")):
                os.remove(os.path.join(self.saving, f"best_epoch_{self.best_epoch}.ckpt"))

            self.best_vloss = loss
            self.best_epoch = self.epoch
    
    def get_saving_path(self) -> Path:
        """ Create depending on hyperparameter and training information model name. """

        label_mode = self.args.diagnosis_label
        dir_model_name = re.compile("([a-zA-Z]+)([0-9]+)").match(self.args.model).group(1)
        self.structure = dir_model_name

        name = f"{str(self.trial_idx) + '_' if self.trial_idx is not None else ''}"\
               f"{self.args.model}_loss:{self.args.loss}_numClass:{config.num_classes}_lr:{np.round(self.args.lr, 4)}_optimizer:{type(self.optimizer).__name__}"\
               f"{'_wd:'+str(np.round(self.args.wd, 4)) if self.args.wd is not None else ''}_sampling:{self.args.do_sampler}_bs:{self.args.batch_size}"\
               f"_weightLoss:{list(np.array(self.weight_loss).astype(int))}_{config.in_imgs_size}_{self.args.diagnosis_label}_seed:{self.args.seed}".replace(" ", "")
        
        if self.args.optimize:
            path = Path(self.args.save_path, label_mode, "optimize", name)
        else:
            path = Path(self.args.save_path, label_mode, dir_model_name, name)

        return path
