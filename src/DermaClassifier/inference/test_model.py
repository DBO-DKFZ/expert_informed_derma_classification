import numpy as np
import torch
import os
import json
from tqdm import tqdm
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC, MulticlassConfusionMatrix, MulticlassF1Score
import DermaClassifier.utils.config as config
from torch.utils.data import DataLoader


class TestSCP2:
    
    def __init__(self, model: list, device: str, save_path: str, num_classes:int=3) -> None:
        """
        Organization to predict for the test and validation setting and compute metrics, so as plots
        of the results.
        
        :model: List of the ensemble model.
        :device: Device info of cpu or gpu.
        :save_path: Path to save the computed metrics information and plots.
        :num_classes: Number of our classes of the multiclass problem.
        """
        self.model = model
        self.device = device

        # Setup all of our metrices
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.auroc = MulticlassAUROC(num_classes=num_classes, average=None)
        self.auroc_average = MulticlassAUROC(num_classes=num_classes)
        self.confmatrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro") 

        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        self.display_labels = ['Invasive\nmelanoma', 'Non-invasive\nmelanoma', 'Nevus']

    def test_step(self, batch: torch.Tensor) -> tuple:
        """ 
        Get the prediction of a single image. 
        
        :batch: Image for which the prediction is computed.
        :return: List with the info of 3D predicted vector, ground truth class, ground truth
            as 3D vector.
        """
        
        img = batch[0].to(self.device)
        label = batch[1]

        # Get the prediciton of each model of the ensemble model
        out_models = []
        for model in self.model:
            output = model(img)
            out_models.append(output)

        # Get the dimension of the highest predicted value 
        pred = torch.Tensor([torch.argmax(torch.mean(torch.stack(out_models).squeeze(), axis=0))]).to(self.device).to(torch.int64)
        target = torch.argmax(label, dim=1)

        # Accuracy
        self.accuracy.update(pred, target)

        # F1
        self.f1.update(pred, target)

        # Confmatrix
        self.confmatrix.update(pred, target)

        return (list(torch.mean(torch.stack(out_models).squeeze(), axis=0).detach().cpu().numpy()), 
                target.detach().cpu().item(),
                list(label.detach().cpu().numpy()[0]))

    def test_step_batch(self, batch: torch.Tensor) -> tuple:
        """ 
        Get the prediction of a batch of images for the lesion. 
        
        :batch: Batch of six images of the lesion for which the prediction is computed.
        :return: List with the info of 3D predicted vector, ground truth class, ground truth
            as 3D vector.
        """
        with torch.no_grad():
            img = batch[0].to(self.device)
            label = batch[1]
            label = torch.mean(label, axis=0).unsqueeze(dim=0)

            out_models = []
            for model in self.model:
                output = model(img)
                out_models.append(torch.mean(output, axis=0).detach().cpu())
                del output 
                torch.cuda.empty_cache()
            out = torch.mean(torch.stack(out_models).squeeze(), axis=0).unsqueeze(dim=0)

            del img
            torch.cuda.empty_cache()
            pred = torch.argmax(out, dim=1)
            target = torch.argmax(label, dim=1)

            # Accuracy
            self.accuracy.update(pred, target)

            # F1
            self.f1.update(pred, target)

            # Confmatrix
            self.confmatrix.update(pred, target)

            return (list(out.detach().cpu().numpy()[0]), 
                    target.detach().cpu().item(),
                    list(label.detach().cpu().numpy()[0]))
    
    def test_batch(self, data: DataLoader, pred:str="single") -> dict:
        """ 
        Iterate over the dataset and compute prediciton for single images or the batch of images to
        predict one class for a lesion.

        :data: dataloader
        :pred: Modus if the prediction is for a single image or the lesion (batch of six images) 
        :return: dictionary of the result metrices
        """

        auroc_t = []
        auroc_o = []

        out_list = []
        label_list = []

        if pred == "single":
            for _, batch in tqdm(enumerate(data)):
                out, target, label = self.test_step(batch)
                auroc_t.append(target)        
                auroc_o.append(out)

                out_list.append(out)
                label_list.append(label)
        elif pred == "batch":
            for _, batch in tqdm(enumerate(data)):
                out, target, label = self.test_step_batch(batch)
                auroc_t.append(target)        
                auroc_o.append(out)

                out_list.append(out)
                label_list.append(label)
        else:
            raise NotImplementedError

        self.auroc.update(torch.tensor(auroc_o), torch.tensor(auroc_t))
        self.auroc_average.update(torch.tensor(auroc_o), torch.tensor(auroc_t))

        # balanced ACC
        m = self.confmatrix.compute()
        bal_acc = sum([m[i, i]/m[i, :].sum() for i in range(m.shape[0])]) / m.shape[0]

        au = self.auroc.compute().cpu().numpy().tolist()
        au.append(self.auroc_average.compute().cpu().numpy().tolist())

        results = {
            "bal_accuracy": bal_acc.cpu().numpy().tolist(),
            "f1": self.f1.compute().item(),
            "accuracy": self.accuracy.compute().cpu().numpy().tolist(),
            "auroc": [au, (auroc_t, auroc_o)],
            "confmatrix": self.confmatrix.compute().cpu().numpy().tolist(),
        }

        self.out_list = out_list
        self.label_list = label_list

        return results
    
    def save_results(self, results: dict, mode: str):
        """ Create confmatrix plot, ROC curve plot and save all metrices in a .json file. """
        
        # Save image of ConfMatrix
        cm = results["confmatrix"]
        _, ax_conf_matrix = plt.subplots(figsize=(9, 7))
        disp = ConfusionMatrixDisplay(np.array(cm), display_labels=self.display_labels)
        disp.plot(ax=ax_conf_matrix, cmap=plt.cm.Blues, values_format='.3g')
        cm_normalized = np.array(cm).astype('float') / np.array(cm).sum(axis=1)[:, np.newaxis]
        ax_conf_matrix.set_xlabel('Predicted labels', fontsize=14, labelpad=10)  # X-Achsen-Beschriftung mit Schriftgröße 14
        ax_conf_matrix.set_ylabel('True labels', fontsize=14, labelpad=10)       # Y-Achsen-Beschriftung mit Schriftgröße 14
        ax_conf_matrix.set_title('{} test dataset'.format({"intern": "Holdout", "extern": "External", 
                                                           "val": "Validation", "demo": "Demo"}[mode]), fontsize=17) 
        plt.tight_layout(pad=2.0)
        
        # Füge Prozentwerte als Text in jede Zelle hinzu
        for i in range(config.num_classes):
            for j in range(config.num_classes):
                ax_conf_matrix.text(j, i, f'\n{cm_normalized[i, j]:.2%}', ha='center', va='top', color="black" if cm[i][j] < (3/4) * np.array(cm).max() else "silver")
        plt.savefig(os.path.join(self.save_path, f"{mode}_confmatrix.png"))

        # Save image of AUROC
        t = results["auroc"][1][0]
        # title = r"$\bf{c}$ Test holdout" if mode == "intern" else r"$\bf{d}$ Test extern"

        average_auroc = np.round(results["auroc"][0][-1], 3)

        _, ax_auroc = plt.subplots(figsize=(9, 7))
        skplt.metrics.plot_roc(t, results["auroc"][1][1], ax=ax_auroc, title="", plot_micro=False)
        plt.legend([f'{l} (area={np.round(results["auroc"][0][l_idx], 3)})' for l_idx, l in enumerate(self.display_labels)] + \
                   [f'Average macro-AUROC (area= {average_auroc})'])
        plt.xlabel("False positive rate", fontsize=14, labelpad=10)
        plt.ylabel("True positive rate", fontsize=14, labelpad=10)
        plt.title('{} test dataset'.format({"intern": "Holdout", "extern": "External", 
                                            "val": "Validation", "demo": "Demo"}[mode]), loc="left", fontsize=17)
        plt.tight_layout(pad=2.0)
        
        plt.savefig(os.path.join(self.save_path, f"{mode}_auroc_.png")) # save as pdf for vector graphics for submission (nature and stuff)

        # Save all data in notebook
        results["auroc"] = results["auroc"][0]

        results["confmatrix_metric"] = self.compute_confmatrix_metric(np.array(results["confmatrix"]))

        with open(os.path.join(self.save_path, f'{mode}_data.json'), 'w+') as f:
            json.dump(results, f, indent=2)
        
    def compute_confmatrix_metric(self, conf_matrix: np.array) -> dict:
        """ Compute per class against the other accuracy, sensitivity, specifity, precision. """
        metrics = dict()
        
        for cl in range(len(conf_matrix)):
            TP = conf_matrix[cl, cl]
            FP = sum(conf_matrix[:, cl]) - TP
            FN = sum(conf_matrix[cl]) - TP
            TN = sum(conf_matrix.diagonal()) - TP

            acc = (TP + TN) / (TP+FP+FN+TN)
            sensitivity = TP / (TP+FN)
            specifity = TN / (FP + TN)
            precision = TP / (TP+FP)

            metrics[f"class_{str(cl)}"] = {
                "accuracy": acc,
                "sensitivity": sensitivity, 
                "specifity": specifity, 
                "precision": precision
            }
    
        return metrics

    def testing(self, data: DataLoader, mode: str, pred:str="single", saving:bool=False):
        """ 
        Compute the predictions and calculate the metrics.
        
        :data: Dataloader of test holdout/external or validation data.
        :mode: Infor which test mode we compute the metrics (holdout/external or validation).
        :pred: Define if predicions is for one image or one lesion (batch of six images).
        :saving: Set true, if confamtrix plot, ROC curves plot and all computed metrices should be saved.
        :return: dictionary of the result metrices (accuracy, ballanced accuracy, F1 score, AUROC values)
        """

        # Reset metrics, so that no old metric is included
        self.auroc.reset()
        self.accuracy.reset()
        self.confmatrix.reset()
        self.f1.reset()
        self.auroc_average.reset()

        results = self.test_batch(data, pred=pred)

        if saving:
            self.save_results(results, mode)
            
        return results
