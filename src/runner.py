import numpy as np
import pandas as pd
import trainers
import hydra
import mlflow
from omegaconf import DictConfig
from pathlib import Path
import utils
import warnings

warnings.filterwarnings("ignore")


def get_trainer(config: DictConfig, model_name: str, X_train, y_train, X_test):
    model_config = config["model"]
    if model_name in model_config.keys():
        if model_name == "catboost":
            trainer = trainers.CatBoostTrainer(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                model_name=config["model"]["catboost"]["name"],
                params=config["model"]["catboost"]["params"],
                num_fold=config["split"]["params"]["n_splits"],
                seeds=config["training"]["seeds"],
            )
        elif model_name == "lightgbm":
            trainer = trainers.LightGBMTrainer(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                params=config["model"]["lightgbm"]["params"],
                num_fold=config["split"]["params"]["n_splits"],
                seeds=config["training"]["seeds"],
            )
    else:
        raise NotImplementedError
    return trainer


def training_step(trainer):
    y_oof, models = trainer.fit()
    y_pred = trainer.predict()
    return y_oof, models, y_pred


def single_target_training(cfg, X_train, y_train, X_test, logger):
    model_name = list(cfg["model"].keys())[0]

    metric = utils.get_metric(cfg)
    metric_name = cfg["metric"]["name"]

    experiment_id = utils.get_experiment_id(cfg)
    run_name = cfg["mlflow"]["run_name"]

    logger.info(f"experiment config: {run_name}")
    logger.info(
        f"CV method: {cfg['split']['name']} {cfg['split']['params']['n_splits']}-Fold"
    )

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        trainer = get_trainer(cfg, model_name, X_train, y_train, X_test)
        y_oof, models, y_pred = training_step(trainer)
        metrics = utils.calc_metrics(cfg, metric, y_train.values, y_oof)
        fig = utils.plot_feature_importance(models, X_train, model_name)

        logger.info(f"CV score : {metrics[metric_name]}")
        utils.mlflow_logger(cfg, metrics, fig, targets=None)

    return y_pred


def multi_target_training(cfg, X_train, y_trains, X_test, logger):
    model_name = list(cfg["model"].keys())[0]

    targets = cfg["training"]["targets"]
    y_oof = np.zeros((len(X_train), len(targets)))
    y_pred = np.zeros((len(X_test), len(targets)))
    metric = utils.get_metric(cfg)
    metric_name = cfg["metric"]["name"]
    figs = []

    experiment_id = utils.get_experiment_id(cfg)
    run_name = cfg["mlflow"]["run_name"]

    logger.info(f"experiment config: {run_name}")
    logger.info(
        f"CV method: {cfg['split']['name']} {cfg['split']['params']['n_splits']}-Fold"
    )

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        for i, target in enumerate(targets):
            logger.info(f"Training for {target}")
            y_train = y_trains[target]
            trainer = get_trainer(cfg, model_name, X_train, y_train, X_test)
            y_oof_, models, y_pred_ = training_step(trainer)
            y_oof[:, i] = y_oof_
            y_pred[:, i] = y_pred_

            fig = utils.plot_feature_importance(models, X_train, model_name, target)
            figs.append(fig)

        metrics = utils.calc_metrics(cfg, metric, y_trains.values, y_oof)
        logger.info(f"CV score : {metrics[metric_name]}")
        utils.mlflow_logger(cfg, metrics, figs, targets)

    return y_pred


def make_submission(config: DictConfig, y_pred):
    targets = config["training"]["targets"]
    output_path = Path(config["globals"]["output_dir"])
    file_name = config["mlflow"]["run_name"]
    submission_df = pd.DataFrame()
    for i, target in enumerate(targets):
        submission_df[target] = y_pred[:, i]
    submission_df.to_csv(output_path / str(file_name + "_sub.csv"), index=False)


@hydra.main(config_path="../config")
def main(cfg: DictConfig):
    logger = utils.get_logger()
    X_train, y_trains, X_test = utils.load_feature(cfg)
    y_pred = multi_target_training(cfg, X_train, y_trains, X_test, logger)
    logger.info("Make submission")
    make_submission(cfg, y_pred)
    logger.info("Finished Training and Prediction!")


if __name__ == "__main__":
    main()
