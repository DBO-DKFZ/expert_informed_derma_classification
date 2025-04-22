import torch
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from DermaClassifier.model.models import create_model
from DermaClassifier.dataset.scp2_dataset import SCP2Dataset
from DermaClassifier.inference.test_model import TestSCP2
import DermaClassifier.utils.config as config
from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC, MulticlassConfusionMatrix, MulticlassF1Score


def test_demo(arg: argparse.ArgumentParser, models: list, device: str, saved_model_dict) -> dict:
    """ 
    Function to predict the results for the demo set. 
    
    :models: List of the loaded ensemble weights.
    :device: Device information to gpu or cpu.
    :saved_model_dict: Information of the loaded weights.
    :return: Computed metrices.
    """
    config.mask = None
    test_set = SCP2Dataset(arg, dataset_type="test", demo=True)
    test = DataLoader(test_set, batch_size=1)

    engine = TestSCP2(model=models, device=device, 
                  save_path=Path("results", "demo", arg.diagnosis_label, saved_model_dict['preprocess']),
                  num_classes=saved_model_dict['num_classes'])
    
    print("Demo set")
    results = engine.testing(test, "demo", pred=arg.pred, saving=arg.saving)
    print("Demo accuracy: ", np.round(results["accuracy"], 4))
    print("Demo f1: ", np.round(results["f1"], 4))
    print("Demo bal_accuracy: ", np.round(results["bal_accuracy"], 4))
    print("Demo auroc: ", np.round(results["auroc"][0], 4))
    print("Demo confmatrix: ", results["confmatrix"])

    return results


def test_patho_panel(arg: argparse.ArgumentParser, models: list, device: str, saved_model_dict) -> dict:
    """ 
    Function to predict the results for test holout, external and validation set. 
    
    :models: List of the loaded ensemble weights.
    :device: Device information to gpu or cpu.
    :saved_model_dict: Information of the loaded weights.
    :return: Computed metrices.
    """
    # Implement testcase
    test_set_intern = SCP2Dataset(arg, dataset_type="test_intern")
    test_set_extern = SCP2Dataset(arg, dataset_type="test_extern")
    arg.diagnosis_label = "majority"
    val_set = SCP2Dataset(arg, dataset_type="val")

    # Set up dataloader
    test_intern = DataLoader(test_set_intern, batch_size=1 if arg.pred == "single" else 6)
    test_extern = DataLoader(test_set_extern, batch_size=1 if arg.pred == "single" else 6)
    val = DataLoader(val_set, batch_size=1 if arg.pred == "single" else 6)

    Path("results").mkdir(parents=True, exist_ok=True)
    engine = TestSCP2(model=models, device=device, 
                      save_path=Path("results", arg.diagnosis_label, saved_model_dict['preprocess']),
                      num_classes=saved_model_dict['num_classes'])

    print("Intern set")
    result_intern = engine.testing(test_intern, "intern", pred=arg.pred, saving=arg.saving)
    print("Intern accuracy: ", np.round(result_intern["accuracy"], 4))
    print("Intern f1: ", np.round(result_intern["f1"], 4))
    print("Intern bal_accuracy: ", np.round(result_intern["bal_accuracy"], 4))
    print("Intern auroc: ", np.round(result_intern["auroc"], 4))

    print("-"*30)
    print("Extern set")
    result_extern = engine.testing(test_extern, "extern", pred=arg.pred, saving=arg.saving)
    print("Extern f1: ", np.round(result_extern["f1"], 4))
    print("Extern accuracy: ", np.round(result_extern["accuracy"], 4))
    print("Extern bal_accuracy: ", np.round(result_extern["bal_accuracy"], 4))
    print("Extern auroc: ", np.round(result_extern["auroc"], 4))

    print("-"*30)
    print("Val set")
    result_val = engine.testing(val, "val", pred=arg.pred, saving=arg.saving)
    print("Extern f1: ", np.round(result_val["f1"], 4))
    print("Extern accuracy: ", np.round(result_val["accuracy"], 4))
    print("Extern bal_accuracy: ", np.round(result_val["bal_accuracy"], 4))
    print("Extern auroc: ", np.round(result_val["auroc"], 4))

    return result_intern, result_extern, result_val


def test_setup(opt: argparse.ArgumentParser) -> dict:
    """ Compute metrices of the predictions for demo or holdout/external/validation.  """
    if torch.cuda.is_available():
        device = f"cuda:{0}"
    else:
        device = "cpu"
    print(f'Device defined as {device}')
    
    # Get all the pathes and files of the ensemble model
    model_dir = [path for path in Path(opt.model).iterdir()] 

    models = []
    args_list = []

    # Load the weights of the ensemble model
    for model_param in model_dir: 

        saved_model_dict = torch.load(model_param)
        assert saved_model_dict['preprocess'] == config.preprocess, 'Wrong preprocess mode in config file!'
        assert saved_model_dict['encoding'] == config.encoding, 'Incorrect encoding mode in config file!'
        args_list.append(saved_model_dict['args'])

        # Create model and load weights
        model = create_model(saved_model_dict['structure'], saved_model_dict['num_classes'], saved_model_dict['args'].diagnosis_label)
        model.to(device)
        model.load_state_dict(saved_model_dict["model_state_dict"])
        model.eval()
        models.append(model)

    arg = saved_model_dict['args']
    
    if opt.demo:
        arg.images_path = "./demo/data"
        arg.pred = "single"
        arg.saving = False
        return test_demo(arg, models, device, saved_model_dict)
    else:
        arg.table_path = opt.table_path
        arg.images_path = opt.images_path
        arg.pred = opt.pred
        arg.saving = opt.saving
        return test_patho_panel(arg, models, device, saved_model_dict)


def test_metrices(db: pd.DataFrame, pred: pd.DataFrame):
    """ Compute metrices for jupyter notebook example of predictions. """
    accuracy = MulticlassAccuracy(num_classes=config.num_classes)
    auroc = MulticlassAUROC(num_classes=config.num_classes, average=None)
    auroc_average = MulticlassAUROC(num_classes=config.num_classes)
    confmatrix = MulticlassConfusionMatrix(num_classes=config.num_classes)
    f1 = MulticlassF1Score(num_classes=config.num_classes, average="macro")

    predicted = torch.Tensor([[float(cl) for cl in pre.strip('][').split(', ')] for pre in db[pred].tolist()])
    target = torch.Tensor([config.encoding[pre] for pre in db.pathoPanel.tolist()]).to(torch.int64)

    for idx in range(len(db)):
        accuracy.update(torch.argmax(predicted[idx]).unsqueeze(dim=0), target[idx].unsqueeze(dim=0))
        confmatrix.update(torch.argmax(predicted[idx]).unsqueeze(dim=0), target[idx].unsqueeze(dim=0))
        f1.update(torch.argmax(predicted[idx]).unsqueeze(dim=0), target[idx].unsqueeze(dim=0))
    
    auroc.update(predicted, target)
    auroc_average.update(predicted, target)
    
    return {"accuracy": accuracy.compute(), 
            "confmatrix": confmatrix.compute(), 
            "f1": f1.compute(),
            "auroc": auroc.compute(),
            "auroc_avg": auroc_average.compute()}
