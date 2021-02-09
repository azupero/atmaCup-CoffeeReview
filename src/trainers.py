import numpy as np
import pandas as pd
import catboost
import lightgbm as lgb
from omegaconf import DictConfig


class CatBoostTrainer:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        model_name: str,
        params: DictConfig,
        num_fold: int,
        seeds: list,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.model_name = model_name
        self.params = params
        self.num_fold = num_fold
        self.seeds = seeds
        self.models = []

    def _build_model(self, model_name: str, params: dict):
        model = catboost.__getattribute__(model_name)(**params)
        return model

    def fit(self):
        oof = np.zeros((len(self.seeds), len(self.y_train)))

        for i, seed in enumerate(self.seeds):
            oof_ = np.zeros((len(self.y_train)))
            params = dict(self.params)
            params["random_state"] = seed  # seed average

            for fold_id in range(self.num_fold):
                tr_idx = self.X_train[self.X_train["fold"] != fold_id].index
                val_idx = self.X_train[self.X_train["fold"] == fold_id].index

                X_train_fold = self.X_train.iloc[tr_idx].drop(columns=["fold"]).values
                X_valid_fold = self.X_train.iloc[val_idx].drop(columns=["fold"]).values

                y_train_fold = self.y_train.iloc[tr_idx].values
                y_valid_fold = self.y_train.iloc[val_idx].values

                model = self._build_model(self.model_name, params)
                model.fit(
                    X_train_fold,
                    y_train_fold,
                    eval_set=[(X_valid_fold, y_valid_fold)],
                    use_best_model=True,
                )

                oof_[val_idx] = model.predict(X_valid_fold)
                self.models.append(model)

            oof[i, :] = oof_
        y_oof = np.mean(oof, axis=0)

        return y_oof, self.models

    def predict(self):
        y_pred = np.mean(
            [model.predict(self.X_test.values) for model in self.models], axis=0
        )
        assert len(y_pred) == len(self.X_test)

        return y_pred


class LightGBMTrainer:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        params: DictConfig,
        num_fold: int,
        seeds: list,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.params = params
        self.num_fold = num_fold
        self.seeds = seeds
        self.models = []

    def fit(self):
        oof = np.zeros((len(self.seeds), len(self.y_train)))

        for i, seed in enumerate(self.seeds):
            oof_ = np.zeros((len(self.y_train)))
            params = dict(self.params)
            params["seed"] = seed  # seed average

            for fold_id in range(self.num_fold):
                tr_idx = self.X_train[self.X_train["fold"] != fold_id].index
                val_idx = self.X_train[self.X_train["fold"] == fold_id].index

                X_train_fold = self.X_train.iloc[tr_idx].drop(columns=["fold"]).values
                X_valid_fold = self.X_train.iloc[val_idx].drop(columns=["fold"]).values

                y_train_fold = self.y_train.iloc[tr_idx].values
                y_valid_fold = self.y_train.iloc[val_idx].values

                train_set = lgb.Dataset(X_train_fold, y_train_fold)
                valid_set = lgb.Dataset(X_valid_fold, y_valid_fold, reference=train_set)

                model = lgb.train(
                    train_set=train_set,
                    valid_sets=[train_set, valid_set],
                    params=params,
                    verbose_eval=100,
                )

                oof_[val_idx] = model.predict(
                    X_valid_fold, num_iteration=model.best_iteration
                )
                self.models.append(model)

            oof[i, :] = oof_
        y_oof = np.mean(oof, axis=0)

        return y_oof, self.models

    def predict(self):
        y_pred = np.mean(
            [model.predict(self.X_test.values) for model in self.models], axis=0
        )
        assert len(y_pred) == len(self.X_test)

        return y_pred
