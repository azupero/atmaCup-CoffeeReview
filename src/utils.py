import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics as skm
import random
import os
import time
import logging
from contextlib import contextmanager
from typing import Optional
import mlflow
from omegaconf import DictConfig
from plotly import express as px
from matplotlib import pyplot as plt
from pathlib import Path


def load_dataset(config: DictConfig):
    input_config = config["globals"]
    input_path = Path(input_config["input_dir"])
    train = pd.read_csv(input_path / "train.csv")
    test = pd.read_csv(input_path / "test.csv")
    submission = pd.read_csv(input_path / "sample_submission.csv")
    return train, test, submission


def save_feature(config: DictConfig, X_train, y_train, X_test):
    feature_config = config["feature"]
    feature_path = Path(feature_config["dir"])
    X_train.to_pickle(feature_path / "X_train.pkl")
    y_train.to_pickle(feature_path / "y_train.pkl")
    X_test.to_pickle(feature_path / "X_test.pkl")


def load_feature(config: DictConfig):
    feature_config = config["feature"]
    feature_path = Path(feature_config["dir"])
    # feature_path = Path(hydra.utils.get_original_cwd())
    # feature_path = feature_path.joinpath("..", "feature")
    X_train = pd.read_pickle(feature_path / "X_train.pkl")
    y_train = pd.read_pickle(feature_path / "y_train.pkl")
    X_test = pd.read_pickle(feature_path / "X_test.pkl")
    return X_train, y_train, X_test


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(out_file=None):
    logger = logging.getLogger()  # loggerの呼び出し
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(message)s]"
    )  # ログ出力の際のフォーマットを定義
    logger.handlers = []  # ハンドラーを追加するためのリスト
    logger.setLevel(logging.INFO)  # ロギングのレベルを設定, 'INFO' : 想定された通りのことが起こったことの確認

    handler = logging.StreamHandler()  # StreamHandler(コンソールに出力するハンドラ)を追加
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # ログをファイルとして出力する際のハンドラ(FileHandler)
    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    logger.info("logger set up")  # "logger set up"を表示
    return logger


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def get_group_k_fold(config: DictConfig, input_df: pd.DataFrame):
    cv_config = config["split"]
    cv_params = cv_config["params"]

    group_col = cv_params["group_col"]
    group_series = input_df[group_col]
    group_key = group_series.unique()

    splitter = KFold(
        n_splits=cv_params["n_splits"],
        shuffle=cv_params["shuffle"],
        random_state=set_seed(cv_params["cv_seed"]),
    )

    for fold_id, (tr_group_idx, val_group_idx) in enumerate(splitter.split(group_key)):
        # tr_group = group_key[tr_group_idx]
        val_group = group_key[val_group_idx]
        # is_tr = group_series.isin(tr_group)
        is_val = group_series.isin(val_group)

        input_df.loc[is_val, "fold"] = fold_id

    input_df["fold"] = input_df["fold"].astype(int)

    return input_df


def get_experiment_id(config: DictConfig):
    tracking_uri = config["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    cli = mlflow.tracking.MlflowClient()
    experiment_id = cli.get_experiment_by_name(experiment_name).experiment_id
    return experiment_id


def mlflow_log_param(config: DictConfig):
    model_config = config["model"]
    # split_config = config["split"]["params"]
    # params = {**params_config, **split_config}

    for model in model_config.keys():
        for key, value in model_config[model]["params"].items():
            mlflow.log_param(model + "_" + key, value)


def mlflow_log_metric(config: DictConfig, metric: np.ndarray):
    metric_config = config["metric"]
    metric_name = metric_config["name"]
    mlflow.log_metric(metric_name, metric)


def mlflow_log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)


def mlflow_log_artifact(config: DictConfig):
    config_path = Path(config["globals"]["config_path"]) / str(
        config["mlflow"]["run_name"] + ".yaml"
    )
    mlflow.log_artifact(config_path)


def mlflow_log_figure(config: DictConfig, fig, target=None):
    figure_config = config["training"]["figure"]
    figure_name = (
        target + "_" + figure_config["name"]
        if target is not None
        else figure_config["name"]
    )
    mlflow.log_figure(fig, "fig/" + figure_name)


def mlflow_logger(config: DictConfig, metrics: dict, figs, targets=None):
    mlflow_log_param(config)
    mlflow_log_metrics(metrics)
    mlflow_log_artifact(config)

    if targets is not None:
        for i in range(len(targets)):
            mlflow_log_figure(config, figs[i], targets[i])
    else:
        mlflow_log_figure(config, figs)


def get_metric(config: DictConfig):
    metric_config = config["metric"]
    metric_name = metric_config["name"]
    metric = skm.__getattribute__(metric_name)
    return metric


def calc_metrics(config: DictConfig, metric, y_true: np.ndarray, y_pred: np.ndarray):
    targets = config["training"]["targets"]
    metric_name = config["metric"]["name"]
    metrics = {}
    finally_score = 0

    if len(targets) == 1:
        score = metric(y_true, y_pred)
        metrics[metric_name] = score
    else:
        for i in range(len(targets)):
            score = metric(y_true[:, i], y_pred[:, i])
            metrics[metric_name + "_" + targets[i]] = score
            finally_score += score
        finally_score = finally_score / len(targets)
        metrics[metric_name] = finally_score
    return metrics


def plot_feature_importance(models, input_df, model_name, target=None):
    columns = input_df.columns.drop(["fold"]).tolist()
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _importance_df = pd.DataFrame()
        _importance_df["feature"] = columns
        if model_name == "catboost":
            importance = model.feature_importances_
        elif model_name == "lightgbm":
            importance = model.feature_importance(importance_type="gain")
        _importance_df["importance"] = importance
        _importance_df["model_no"] = i
        feature_importance_df = pd.concat(
            [feature_importance_df, _importance_df], axis=0
        )
        del _importance_df

    order = (
        feature_importance_df.groupby(["feature"])[["importance"]]
        .sum()
        .sort_values(by="importance", ascending=False)
        .index.tolist()[:50]
    )

    fig = px.box(
        feature_importance_df.query("feature in @order"),
        x="importance",
        y="feature",
        category_orders={"feature": order},
        width=1250,
        height=600,
        title=f"target:{target} feature importance"
        if target is not None
        else "feature importance",
    )
    return fig
