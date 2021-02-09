import pandas as pd
import utils
import preprocess
import category_encoders as ce
from sklearn.model_selection import KFold
from xfeat import TargetEncoder
from tqdm import tqdm
import hydra
from omegaconf import DictConfig


class BaseBlock(object):
    def fit(self, input_df, y=None):
        return self.transform(input_df)

    def transform(self, input_df):
        return NotImplementedError()


class OrdinalEncodingBlock(BaseBlock):
    def __init__(self, cols):
        self.cols = cols
        self.encoder = None

    def fit(self, input_df, y=None):
        self.encoder = ce.OrdinalEncoder(handle_unknown="value", handle_missing="value")
        self.encoder.fit(input_df[self.cols])

    def transform(self, input_df):
        return self.encoder.transform(input_df[self.cols]).add_suffix("_OE")


class CountEncodingBlock(BaseBlock):
    def __init__(self, cols):
        self.cols = cols
        self.encoder = None

    def fit(self, input_df, y=None):
        self.encoder = ce.CountEncoder(handle_unknown=-1, handle_missing="count")
        self.encoder.fit(input_df[self.cols])

    def transform(self, input_df):
        return self.encoder.transform(input_df[self.cols]).add_suffix("_CE")


class OneHotEncodingBlock(BaseBlock):
    def __init__(self, cols):
        self.cols = cols
        self.encoder = None

    def fit(self, input_df, y=None):
        self.encoder = ce.OneHotEncoder(
            use_cat_names=True, handle_unknown="value", handle_missing="value"
        )
        self.encoder.fit(input_df[self.cols])

    def transform(self, input_df):
        return self.encoder.transform(input_df[self.cols]).add_suffix("_OHE")


class GroupingBlock(BaseBlock):
    def __init__(self, cat_cols: list, target_cols: list, methods: list):
        self.cat_cols = cat_cols
        self.target_cols = target_cols
        self.methods = methods

        self.df = None
        self.a_cat = None

    def fit(self, input_df, y=None):
        self.df = [self._agg(input_df, target_col) for target_col in self.target_cols]
        self.df = pd.concat(self.df, axis=1)
        self.df[self.cat_cols] = self.a_cat[self.cat_cols]

    def transform(self, input_df):
        output_df = pd.merge(
            input_df[self.cat_cols], self.df, on=self.cat_cols, how="left"
        )
        output_df = output_df.drop(columns=self.cat_cols, axis=1)
        return output_df

    def _agg(self, input_df, target_col):
        _df = input_df.groupby(self.cat_cols, as_index=False).agg(
            {target_col: self.methods}
        )
        cols = self.cat_cols + [
            f"agg_{method}_{'_and_'.join(self.cat_cols)}_by_{target_col}"
            for method in self.methods
        ]
        _df.columns = cols
        self.a_cat = _df[self.cat_cols]
        return _df.drop(columns=self.cat_cols, axis=1)


class TargetEncodingBlock(BaseBlock):
    def __init__(self, cols, group_cols, target, splitter):
        self.cols = cols
        self.group_cols = group_cols if group_cols is not None else None
        self.target = target
        self.splitter = splitter if splitter is not None else None
        self.encoder = None

    def fit_transform(self, input_df, y):
        input_df = input_df.copy()
        num_fold = input_df["fold"].nunique()
        output_df = pd.DataFrame()

        if self.group_cols:
            input_df, group_keys = self._grouping(input_df, self.group_cols)
            df_cols = input_df.columns.append(y.columns)
            self.cols.extend(group_keys)
        else:
            df_cols = input_df.columns.append(y.columns)

        self.encoder = TargetEncoder(
            input_cols=self.cols,
            target_col=self.target,
            fold=self.splitter,
            output_suffix=f"_TE_{self.target}",
        )

        for fold_id in range(num_fold):
            X_train_fold = input_df[input_df["fold"] != fold_id]
            X_valid_fold = input_df[input_df["fold"] == fold_id]

            y_train_fold = y.iloc[X_train_fold.index]
            y_valid_fold = y.iloc[X_valid_fold.index]

            _train_df = pd.concat([X_train_fold, y_train_fold], axis=1)
            _valid_df = pd.concat([X_valid_fold, y_valid_fold], axis=1)

            self.encoder.fit_transform(_train_df)
            _valid_df = self.encoder.transform(_valid_df)

            output_df = pd.concat([output_df, _valid_df], axis=0)

        output_df = output_df.sort_index()
        output_df = output_df.drop(columns=df_cols)

        return output_df

    def transform(self, X_train, y_train, X_test):
        X_train = X_train.copy()
        X_test = X_test.copy()

        if self.group_cols:
            X_train, _ = self._grouping(X_train, self.group_cols)
            X_test, group_keys = self._grouping(X_test, self.group_cols)
            _train_df = pd.concat([X_train, y_train], axis=1)
            df_cols = X_test.columns
            self.cols.extend(group_keys)
        else:
            _train_df = pd.concat([X_train, y_train], axis=1)
            df_cols = X_test.columns

        self.encoder = TargetEncoder(
            input_cols=self.cols,
            target_col=self.target,
            fold=self.splitter,
            output_suffix=f"_TE_{self.target}",
        )

        self.encoder.fit_transform(_train_df)

        output_df = self.encoder.transform(X_test)
        output_df = output_df.drop(columns=df_cols)

        return output_df

    def _grouping(self, input_df, group_cols):
        group_keys = []
        for cols in group_cols:
            key = "grouping_" + "_and_".join([col for col in cols])
            if len(cols) == 2:
                input_df[key] = (
                    input_df[cols[0]].astype(str) + "_" + input_df[cols[1]].astype(str)
                )
            elif len(cols) == 3:
                input_df[key] = (
                    input_df[cols[0]].astype(str)
                    + "_"
                    + input_df[cols[1]].astype(str)
                    + "_"
                    + input_df[cols[2]].astype(str)
                )
            group_keys.append(key)

        return input_df, group_keys


def get_count_encoding_features(config: DictConfig, train_df, test_df):
    ce_config = config["feature"]["processes"]["count_encoding"]
    cols = ce_config["cols"]

    encoder = CountEncodingBlock(cols=cols)
    encoder.fit(train_df)
    encoded_train_df = encoder.transform(train_df)
    encoded_test_df = encoder.transform(test_df)
    # output_df = pd.concat([encoded_train_df, encoded_test_df], axis=0)
    return encoded_train_df, encoded_test_df


def get_ordinal_encoding_features(config: DictConfig, train_df, test_df):
    oe_config = config["feature"]["processes"]["ordinal_encoding"]
    cols = oe_config["cols"]

    encoder = OrdinalEncodingBlock(cols=cols)
    encoder.fit(train_df)
    encoded_train_df = encoder.transform(train_df)
    encoded_test_df = encoder.transform(test_df)
    # output_df = pd.concat([encoded_train_df, encoded_test_df], axis=0)
    return encoded_train_df, encoded_test_df


def get_onehot_encoding_features(config: DictConfig, train_df, test_df):
    ohe_config = config["feature"]["processes"]["onehot_encoding"]
    cols = ohe_config["cols"]

    encoder = OneHotEncodingBlock(cols=cols)
    encoder.fit(train_df)
    encoded_train_df = encoder.transform(train_df)
    encoded_test_df = encoder.transform(test_df)
    # output_df = pd.concat([encoded_train_df, encoded_test_df], axis=0)
    return encoded_train_df, encoded_test_df


def get_target_encoding_features(config: DictConfig, train_df, test_df, y):
    te_config = config["feature"]["processes"]["target_encoding"]
    cols = te_config["cols"]
    group_cols = (
        te_config["group_cols"] if te_config["group_cols"] is not None else None
    )
    targets = config["training"]["targets"]
    splitter = KFold(
        n_splits=te_config["splitter"]["params"]["n_splits"],
        shuffle=te_config["splitter"]["params"]["shuffle"],
        random_state=utils.set_seed(te_config["splitter"]["params"]["seed"]),
    )

    encoded_train_df = pd.DataFrame()
    encoded_test_df = pd.DataFrame()

    for target in targets:
        y_target = y[[target]]
        encoder = TargetEncodingBlock(cols, group_cols, target, splitter)
        _encoded_train_df = encoder.fit_transform(train_df, y_target)
        _encoded_test_df = encoder.transform(train_df, y_target, test_df)

        # output_tmp = pd.concat([encoded_train_df, encoded_test_df], axis=0)
        encoded_train_df = pd.concat([encoded_train_df, _encoded_train_df], axis=1)
        encoded_test_df = pd.concat([encoded_test_df, _encoded_test_df], axis=1)

    return encoded_train_df, encoded_test_df


def get_altitude_features(train_df, test_df, config: DictConfig = None):
    cols = [
        "altitude_low_meters",
        "altitude_high_meters",
        "altitude_mean_meters",
    ]

    output_train_df = train_df[cols].copy()
    output_test_df = test_df[cols].copy()

    return output_train_df, output_test_df


def get_numberof_bags_feature(train_df, test_df, config: DictConfig = None):
    cols = [
        "numberof_bags",
    ]

    output_train_df = train_df[cols].copy()
    output_test_df = test_df[cols].copy()

    return output_train_df, output_test_df


def get_bag_weight_feature(train_df, test_df, config: DictConfig = None):
    cols = [
        "bag_weight_g",
        "bag_weight_log_g",
    ]

    output_train_df = train_df[cols].copy()
    output_test_df = test_df[cols].copy()

    return output_train_df, output_test_df


def get_hervest_year_features(train_df, test_df, config: DictConfig = None):
    cols = [
        "harvest_year",
    ]

    output_train_df = train_df[cols].copy()
    output_test_df = test_df[cols].copy()

    return output_train_df, output_test_df


def get_grading_date_features(train_df, test_df, config: DictConfig = None):
    cols = [
        "grading_date_raw",
        "grading_date_julian",
        "grading_date_year",
        "grading_date_month",
        "grading_date_day",
    ]

    output_train_df = train_df[cols].copy()
    output_test_df = test_df[cols].copy()

    return output_train_df, output_test_df


def get_expiration_features(train_df, test_df, config: DictConfig = None):
    cols = [
        "expiration_raw",
        "expiration_julian",
        "expiration_year",
        "expiration_month",
        "expiration_day",
    ]

    output_train_df = train_df[cols].copy()
    output_test_df = test_df[cols].copy()

    return output_train_df, output_test_df


def get_moisture_features(train_df, test_df, config: DictConfig = None):
    cols = [
        "moisture",
    ]

    output_train_df = train_df[cols].copy()
    output_test_df = test_df[cols].copy()

    return output_train_df, output_test_df


def get_defects_features(train_df, test_df, config: DictConfig = None):
    cols = [
        # "category_one_defects",
        # "category_two_defects",
        # "category_sum_defects",
        "is_specialty",
    ]

    output_train_df = train_df[cols].copy()
    output_test_df = test_df[cols].copy()

    return output_train_df, output_test_df


def get_quakers_features(train_df, test_df, config: DictConfig = None):
    cols = [
        "quakers",
    ]

    output_train_df = train_df[cols].copy()
    output_test_df = test_df[cols].copy()

    return output_train_df, output_test_df


def get_cross_num_features(train_df, test_df, config: DictConfig = None):
    cols = [
        "yield",
        "log_yield",
    ]

    output_train_df = train_df[cols].copy()
    output_test_df = test_df[cols].copy()

    return output_train_df, output_test_df


def get_agg_base_features(config: DictConfig, train_df, test_df):
    cat_cols = list(config["cat_cols"])
    target_cols = list(config["target_cols"])
    methods = list(config["methods"])

    encoder = GroupingBlock(cat_cols=cat_cols, target_cols=target_cols, methods=methods)
    encoder.fit(train_df)
    encoded_train_df = encoder.transform(train_df)
    encoded_test_df = encoder.transform(test_df)
    # output_df = pd.concat([encoded_train_df, encoded_test_df], axis=0)

    return encoded_train_df, encoded_test_df


def get_agg_region_features(config: DictConfig, train_df, test_df):
    agg_config = config["feature"]["processes"]["agg_region"]
    output_train_df, output_test_df = get_agg_base_features(
        agg_config, train_df, test_df
    )

    return output_train_df, output_test_df


def get_agg_company_features(config: DictConfig, train_df, test_df):
    agg_config = config["feature"]["processes"]["agg_company"]
    output_train_df, output_test_df = get_agg_base_features(
        agg_config, train_df, test_df
    )

    return output_train_df, output_test_df


def to_features(config: DictConfig, train_df, test_df, logger):
    feature_config = config["feature"]["processes"]
    processes = feature_config["to_features"]
    targets = config["training"]["targets"]

    logger.info(f"All processing method : {processes}")

    encoding_processes = [process for process in processes if "encoding" in process]
    other_processes = list(set(processes) - set(encoding_processes))

    y = train_df[targets]
    train_df = train_df.drop(columns=targets)
    fold_df = train_df["fold"].copy()

    output_train_df = pd.DataFrame()
    output_test_df = pd.DataFrame()

    # category encoding process
    logger.info("Category encoding")
    for func in tqdm(encoding_processes):
        logger.info(f"Now processing method : {func}")
        func = globals().get(func)
        if func.__name__ == "get_target_encoding_features":
            _train, _test = func(config, train_df, test_df, y)
        else:
            _train, _test = func(config, train_df, test_df)
        assert len(_train) == len(train_df), func.__name__
        assert len(_test) == len(test_df), func.__name__
        output_train_df = pd.concat([output_train_df, _train], axis=1)
        output_test_df = pd.concat([output_test_df, _test], axis=1)

    # merge encoded feature to input
    train_df = pd.concat([train_df, output_train_df], axis=1)
    test_df = pd.concat([test_df, output_test_df], axis=1)

    # other process
    logger.info("Other feature processing")
    if len(other_processes) != 0:
        for func in tqdm(other_processes):
            logger.info(f"Now processing method : {func}")
            func = globals().get(func)
            _train, _test = func(config=config, train_df=train_df, test_df=test_df)
            assert len(_train) == len(train_df), func.__name__
            assert len(_test) == len(test_df), func.__name__
            output_train_df = pd.concat([output_train_df, _train], axis=1)
            output_test_df = pd.concat([output_test_df, _test], axis=1)
    else:
        logger.info("No other feature processing")

    # merge fold column to train
    output_train_df = pd.concat([output_train_df, fold_df], axis=1)

    return output_train_df, output_test_df, y


@hydra.main(config_path="../config")
def main(cfg: DictConfig):
    logger = utils.get_logger()
    # load dataset
    logger.info("Loading dataset")
    train, test, submission = utils.load_dataset(cfg)
    # preprocessing
    logger.info("Preprocessing")
    train, test = preprocess.to_preprocess(cfg, train, test, logger)
    # make fold
    logger.info("Make fold for CV")
    train = utils.get_group_k_fold(cfg, train)
    # get features
    logger.info("Make features")
    X_train, X_test, y_train = to_features(cfg, train, test, logger)
    # save features
    logger.info(
        f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}"
    )
    logger.info("Save features")
    utils.save_feature(cfg, X_train, y_train, X_test)
    logger.info("Finished feature engineering!")


if __name__ == "__main__":
    main()
