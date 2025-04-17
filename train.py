import argparse
import os
import torch
import optuna
import joblib

import DermaClassifier.utils.config as config
import DermaClassifier.utils.train_setup as train_setup
from DermaClassifier.utils.utils import get_saving_path, fix_randomness


def main(opt: argparse.ArgumentParser):

    # If we want to optimize the model with optuna
    if opt.optimize:
        pruner = optuna.pruners.NopPruner()

        study = optuna.create_study(direction="maximize",
                                    storage=config.local_path,
                                    study_name=opt.project_name,
                                    load_if_exists=True,
                                    pruner=pruner)

        study.optimize(lambda trial: train_setup.objective(trial, opt), n_trials=opt.trials, timeout=None)

        path = get_saving_path(opt)
        joblib.dump(study, os.path.join(path, "study" +".pkl"))
        trials_df = study.trials_dataframe()
        trials_df.to_csv(os.path.join(path, "trials" +".csv"))
    
    # Standard training with given predefined parameter
    else:
        opt.do_sampler = config.do_sampler
        train_setup.train(opt, weight_loss=config.weight_loss)

    print("Finished with training process.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data definition
    parser.add_argument("--save_path", type=str, default="./runs/")

    # Training settings
    parser.add_argument("--continue_train", type=bool, default=False, help="Continue the training!")
    parser.add_argument("--project_name", type=str, default="optimize_derma_classifier")

    # Optimization parameter
    parser.add_argument("--optimize", type=bool, default=False)
    parser.add_argument("--trials", type=int, default=10)

    # Define hyperparameter
    parser.add_argument("--model", type=str, default="efficientnetB2", help="Define which architecture.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--loss", type=str, default="ce", help="[ce, mse, kl]")

    # Configuration
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", type=bool, default=False)

    args = parser.parse_args()
    fix_randomness(args.seed)

    # Hyperparameter
    args.lr = config.lr
    args.wd = config.wd
    args.sampling = config.sampling
    args.batch_size = config.bs
    args.diagnosis_label = config.diagnosis_label

    # Path definitions
    args.table_path = os.path.join(config.data_path, config.patho_panel_tabel)
    args.images_path = os.path.join(config.data_path, f"{config.in_imgs_size}x{config.in_imgs_size}")
    
    if (not os.path.exists(args.table_path) or not os.path.exists(args.images_path)):
        raise FileNotFoundError("Data cannot be found!")

    main(args)

    if torch.cuda.is_available():
        with torch.no_grad():
            torch.cuda.empty_cache()
