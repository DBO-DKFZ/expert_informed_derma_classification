# Expert-informed dermatoscopic classifier

This repository provides code to test, train, and tune a multi-classification network for dermatoscopic RGB image analysis, and accompanies the paper:

Expert-informed melanoma classification with dermoscopic and histopathologic data: Results and recommendations for practice, Haggenmüller, S.; Heinlein, L.; Abels, J.; et al., Conference/Journal TBD, 2025

This code is developed and maintained by [Abels, J](https://github.com/JuAbels).

The code for the histopathological classifier can be found [here](https://github.com/DBO-DKFZ/expert_informed_histo_classification).

In the following section, we provide installation instructions, a demonstration using an example dataset, and guidance for training and evaluation with our dataset.

The model weights referenced in the paper are located in the [`weights`](./weights) directory.
Weights trained using the one-hot encoded majority vote variant are stored in [`weights/ohe/rgb_darker`](./weights/ohe/rgb_darker/), while those trained with soft-labels to incorporate uncertainty are stored in [`weights/sl/rgb_darker`](./weights/sl/rgb_darker).
All training configurations and setup information are provided in the configuration file [`src/DermaClassifier/utils/config.py`](src/DermaClassifier/utils/config.py), which can be modified as needed.
This repository includes only the model weights corresponding to the approach described in the paper, which features a preprocessing step that darkens the lesion.

## Pipeline overview
![Dermoscopic classifier training](./figures/ensemble_model.png)

## System Requirements
The Code was tested on Debian GNU/Linux 12 system with GPU (Nvidia V100 SXM2 15.7G) using Python 3.11.
All dependencies are listed in `requirements.txt`.

## Installation guide

**Note**: an active internet connection is required.

Clone this repository and navigate to the project directory:
```
# Clone the repository
git clone git@github.com:DBO-DKFZ/expert_informed_histo_classification.git

# Navigate into the directory
cd expert_informed_histo_classification
```
If Python 3.11 and `venv` are not already installed, you can install them via `apt`:
```
sudo apt install python3.11 python3.11-venv
```
Then, run the following commands to create and activate the environment, and to install all the required packages:
```
# Create a new venv environment
python3.11 -m venv venv

# Activate the environment
. venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

## Demo

To demonstrate the model training process, 30 images from the HAM10K dataset are included.
An overview of the demo data can be found in the table [`data/demo_label_data.csv`](data/demo_label_data.csv).
In addition to the original diagnosis from the dataset, we included randomly selected diagnoses labeled as *softlabel1* through *softlabel8* to demonstrate our soft-label approach for handling uncertainty.

### Jupyter Notebook

The Jupyter notebook [`src/demo.ipynb`](src/demo.ipynb) showcases our training, optimization, and evaluation process using the demo dataset.
Additionally, it reproduces the model prediction metrics described in the paper and supplementary materials for both the holdout and external test datasets.
To proceed, activate the environment, start Jupyter Notebook, and navigate to the file [`src/demo.ipynb`](src/demo.ipynb):
```
# Activate the environment if not activated
. venv/bin/activate

# Start jupyter notebook
jupyter notebook
```

### Testing via console

To test predictions using the demo test set with our model weights from the command line, execute: 
```
python src/test.py --demo True
```
The metrics in  the console output will appear as follows:
```
Demo accuracy:  0.3333
Demo f1:  0.1905
Demo bal_accuracy:  nan
Demo auroc:  [0.5    0.5    0.625  0.5417]
Demo confmatrix:  [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 2.0]]
```
## Reproducing paper results

The directory [`predictions/all_predictions`](./predictions/all_predictions) contains all model predictions using different label encoding variants (majority vote and soft-labels), and preprocessing methods (darker lesion, contrast-enhanced lesion, grayscale image, and normal image).
These results correspond to the validation, holdout, and external test sets, as described in the paper and its supplementary materials, and can be used to reproduce all statistical measurements presented.

The [`predictions/paper`](./predictions/paper/) directory contains predictions generated with the model weights using the darker lesion preprocessing step.
These are intended to reproduce the paper’s confusion matrices and ROC curves.

### Statistic

To reproduce the statistical results from the paper, run the following:
```
python src/results.py --result_case statistic --input_path predictions/all_predictions
```

### Plots

To reproduce the plots from the paper, run the following:
```
# Create plot with confmatrix
python src/results.py --result_case confusion --input_path predictions/paper

# Create plot with ROC curves
python src/results.py --result_case auroc --input_path predictions/paper
```


## Training with the data of SCP

To perform new training, optimization, or evaluation with our dataset, follow the instructions below.
You will need both the dataset and the pathological panel table.

In the [`src/DermaClassifier/utils/config.py`](src/DermaClassifier/utils/config.py) file, you can specify the preprocessing step for training your model by setting the `preprocess` variable (line 11).
The `encode_label` variable (line 12) allows you to choose between majority vote with one-hot encoding or incorporating uncertainty with soft-labels.

Additionally, you need to define the name of the pathological panel table using the `patho_panel_tabel` variable (line 15), and specify the directory containing the images, the pathological panel and the data splits using the `data_path` variable (line 16).

### Training

![Dermoscopic classifier training](./figures/training_process.png)

All hyperparameters are defined in [`src/DermaClassifier/utils/hyperparmeter.py`](src/DermaClassifier/utils/hyperparmeter.py) for all preprocessing and diagnosis encoding variants.
The following attributes allow configuring the training:
* `model`: Defines the model architecture (default `efficientnetB2`). Please refer to the `create_model` function in [`src/DermaClassifier/model/models.py`](src/DermaClassifier/model/models.py) for other available model architectures.
* `epochs`: Defines the number of epochs for the training.
* `loss`: Specifies the loss function. Available are Cross-Entropy (`ce`, default), Mean Squared Error (`mse`), L1 Loss (`l1`), and KL Divergence (`kl`).
* `seed`: Defines the seed to allow for reproducibility. For our ensemble model, we used the following seeds: `0`, `1`, `42` (default), `73`, `123`, `140` and `2024`.

Trained model weights are saved in the `runs` directory, with subdirectories and filenames generated automatically based on the chosen setup.

A training run can be started using e.g. the following command:
```
python src/train.py --model efficientnetB2 --epochs 10 --loss ce --seed 42
```

### Optimization

To start an optimization run, set the `--optimize` attribute to `True`.
The `--trials` argument defines how many different hyperparameter configurations are tested.
Similar to the training run, a directory is automatically created based on the preprocessing method and diagnosis encoding.
Within this directory, the weights for each trial are saved along with a table file listing all tested hyperparameter configurations. 

The tested hyperparameters including detailed information of their search process are stored in an Optuna database file named `optuna.db`.
A fixed seed ensures reproducibility of the optimization, for which we used the seed `42`.

An optimization run can be started using e.g. the following command:
```
python src/train.py --model efficientnetB2 --epochs 10 --loss ce --optimize True --trials 5 --project_name test_opti
```

### Testing

To assess the performance of the trained model, use the `--pred` attribute to specify whether the evaluation should be performed on a `single` image, or on a `batch` of six images per lesion. 
* `--pred single`: Generates a prediction for each dermatoscopic image in the dataset.
* `--pred batch`: Outputs a prediction for a batch of six images of a lesion from the dataset.

To save all statistics and plots in a `results` directory, set the `--saving` attribute to `True`. 

Adjust the `args.model` variable (line 30) to point to the directory of your newly trained model.

A testing run can be started using e.g. the following command:
```
python src/test.py --pred batch --saving True
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
