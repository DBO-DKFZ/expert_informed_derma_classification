import torch
import optuna
import argparse
from torch.utils.data import DataLoader

import DermaClassifier.utils.config as config
from DermaClassifier.model.models import create_model
from DermaClassifier.dataset.scp2_dataset import SCP2Dataset
from DermaClassifier.trainers.trainer import Trainer
from DermaClassifier.utils.utils import get_device, fix_randomness


def train(opt: argparse.ArgumentParser, weight_loss: torch.Tensor, trial_idx: int=None):
    """
    Call of the training logic.

    :opt: Information of paths, training mode and hyperparameter.
    :weight_loss: Weight information of the classes.
    :trial_idx: Index of optimization run.
    :return: Training class, include information of runs.
    """
    
    fix_randomness(opt.seed)

    assert type(opt.do_sampler) is bool

    # Get device > cpu or gpu
    device = get_device()
    
    # Create training and validation set
    train_set = SCP2Dataset(opt, dataset_type="train", demo=opt.demo)
    val_set = SCP2Dataset(opt, dataset_type="val", demo=opt.demo)
    train = DataLoader(train_set, batch_size=opt.batch_size,
                       sampler=config.sampler(train_set, opt.sampling, seed=opt.seed)) if opt.do_sampler \
        else DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
    val = DataLoader(val_set, batch_size=1)

    if not opt.demo:
        assert set(val_set.slideId).intersection(set(train_set.slideId)) == set(), "SlideId in val and train!"
    
    # Create model
    model = create_model(opt.model, config.num_classes, opt.diagnosis_label)
    model.to(device)

    optimizer = config.optimize(model.parameters(), opt.lr, opt.wd)
    training = Trainer(opt, device, model, optimizer, train, val,
                       weight_loss=weight_loss, trial_idx=trial_idx)

    # Start training
    training.train()

    return training


def objective(trial: optuna.trial.Trial, opt: argparse.ArgumentParser) -> float:
    """
    Optuna logic to perform the optimization.
    :trial: information to try new hyperparameter setup.
    :opt: Path and training information.
    :return: AUROC value, which should be maximized.
    """
    lr = trial.suggest_float("lr", 0.0001, 0.001) 
    weight_decay = trial.suggest_float("wd", 0.0, 1e-4)
    do_sampler = False if opt.demo else trial.suggest_categorical("do_sampling", ["yes", "no"])
    sampling = trial.suggest_float("sampling", 0.4, 1.0)
    batch_size = trial.suggest_int("batch_size", 1, 5) if opt.demo else trial.suggest_int("batch_size", 8, 24)
    weight_cl1 = trial.suggest_int("weight_melanom", 1, 3)
    weight_cl2 = trial.suggest_int("weight_insitu", 1, 5)

    structure = "efficientnetB2"

    opt.lr = lr
    opt.batch_size = batch_size
    opt.model = structure
    opt.wd = weight_decay
    opt.sampling = do_sampler if do_sampler == "no" else sampling
    opt.do_sampler = True if do_sampler == "yes" else False
    print(f"PARAMETER: {opt.batch_size} | {opt.lr} | {weight_decay} | {opt.model} | {sampling}")

    # start training with the new choosen hyperparameters
    training = train(opt, weight_loss=torch.Tensor([weight_cl1, weight_cl2, 1]), trial_idx=trial.number)

    return training.auroc_val.compute().item()
