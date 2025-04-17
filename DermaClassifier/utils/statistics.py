import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import auroc, accuracy, f1_score

RANDOM_STATE = 42


def ci(y_true: pd.DataFrame, y_pred: pd.DataFrame, patient_ids: pd.DataFrame, function: Callable, class_wise: bool = False, n_bootstrap: int = 1000) -> str:
    """
    Calculates the 95% CI for given predictions based on the metric provided via function. Bootstraps/samples based on
    patients, and then gathers respective lesions.

    :param y_true: true labels, index needs to be consistent with y_pred and patient_ids
    :param y_pred: predictions, index needs to be consistent with y_true and patient_ids
    :param patient_ids: the patient ids, index needs to be consistent with y_true and y_pred
    :param function: the function to calculate the metric. It needs to handle the input in the form of a dataframe (with columns y_true, y_pred)
    :param class_wise: whether the function calculates class-wise values
    :param n_bootstrap: the number of bootstraps
    :return: the CI as a string in the form of '.3f-.3f', or '[.3f-.3f, .3f-.3f, ...]' for class-wise results
    """
    # merge all data into one dataframe (for sampling)
    tmp = pd.merge(pd.merge(y_true, y_pred, left_index=True, right_index=True), patient_ids, left_index=True, right_index=True)

    # get unique patients
    unique_patients = pd.Series(patient_ids.unique())

    # bootstrap and accumulate all metric results
    result = []
    for i in range(n_bootstrap):
        sampled_patients = unique_patients.sample(n=len(unique_patients), replace=True, random_state=RANDOM_STATE+i)

        # count how many patients are sampled
        counts = sampled_patients.value_counts()
        counts.name = "count"
        merged = tmp.merge(counts, left_on='randomId1', right_index=True)

        # repeat rows depending on counts
        sampled_df = merged.loc[merged.index.repeat(merged['count'])].drop(columns='count')

        metric = function(sampled_df.drop(columns='randomId1'))

        result.append(metric)

    # convert to pd.Series
    result = pd.Series(result)

    # calculate 95% CIs
    if class_wise:
        ret = '['
        for idx in range(len(result[0])):
            out = result.apply(lambda x: x[idx]).quantile([0.025, 0.975])
            ret += f'{out.iloc[0]:.3f}-{out.iloc[1]:.3f}, '

        ret += ']'
    else:
        out = result.quantile([0.025, 0.975])
        ret = f'{out.iloc[0]:.3f}-{out.iloc[1]:.3f}'

    return ret


def statistic_calculation(filenames):
    """
    :filenames:
    """
    
    function_clswise = lambda x: auroc(torch.tensor(np.stack(x.iloc[:, 1].tolist())), torch.tensor(x.iloc[:, 0].values, dtype=torch.int), num_classes=3, average=None, task='multiclass')
    functions = {'AUROC': lambda x: auroc(torch.tensor(np.stack(x.iloc[:, 1].tolist())), torch.tensor(x.iloc[:, 0].values, dtype=torch.int), num_classes=3, average='macro', task='multiclass'),
                'Acc.': lambda x: accuracy(torch.tensor(np.stack(x.iloc[:, 1].tolist())), torch.tensor(x.iloc[:, 0].values, dtype=torch.int), num_classes=3, average='macro', task='multiclass'),
                'F1': lambda x: f1_score(torch.tensor(np.stack(x.iloc[:, 1].tolist())), torch.tensor(x.iloc[:, 0].values, dtype=torch.int), num_classes=3, average='macro', task='multiclass')}

    filenames = filenames.rglob('*.csv')

    for file in filenames:
        print(f'Current file: {file}')
        df = pd.read_csv(file)

        patient_ids = df['randomId1']

        # convert str labels to numbers (2 steps because replace down casting will be deprecated soon)
        df['pathoPanel'] = df['pathoPanel'].replace({'invasive melanoma': '0', 
                                                    'non-invasive melanoma': '1', 'nevus': '2'}).astype(int)

        for classifier in df.columns.drop(['randomId1', 'pathoPanel']):  # pathoPanel > target
            if classifier == "randomId2":
                continue
            print(f'Classifier: {classifier}')
            # convert list strings to proper lists
            df[classifier] = df[classifier].apply(json.loads)

            for key, func in functions.items():
                print(f"{key}: {func(pd.merge(df['pathoPanel'], df[classifier], left_index=True, right_index=True))}, "
                    f"95% CI {ci(y_true=df['pathoPanel'], y_pred=df[classifier], patient_ids=patient_ids, function=func)}")

            print(f"AUROC-multiclass: {function_clswise(pd.merge(df['pathoPanel'], df[classifier], left_index=True, right_index=True))}, "
                f"95% CI {ci(y_true=df['pathoPanel'], y_pred=df[classifier], patient_ids=patient_ids, function=function_clswise, class_wise=True)}")
        
            print("-" * 40)
        
        print("#"* 100)
