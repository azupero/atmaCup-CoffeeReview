import numpy as np
import pandas as pd
import utils
from tqdm import tqdm
from omegaconf import DictConfig


def uniform_bag_weight(input_df: pd.DataFrame):
    output_df = input_df.copy()
    tmp_df = input_df["bag_weight"].str.extract(r"(\d*)\s*([a-zA-Z]*)")
    tmp_df.columns = ["bag_weight_value", "bag_weight_unit"]

    tmp_df["bag_weight_value"] = tmp_df["bag_weight_value"].astype(int)

    tmp_1 = tmp_df[["bag_weight_value"]][tmp_df["bag_weight_unit"] == "kg"].copy()
    tmp_1["bag_weight_value"] = tmp_1["bag_weight_value"] * 1000

    tmp_2 = tmp_df[["bag_weight_value"]][tmp_df["bag_weight_unit"] == "lbs"].copy()
    tmp_2["bag_weight_value"] = tmp_2["bag_weight_value"] * 453

    tmp_3 = tmp_df[["bag_weight_value"]][tmp_df["bag_weight_unit"] == "kg"].copy()
    tmp_3["bag_weight_value"] = tmp_3["bag_weight_value"] * 1000

    tmp_df.loc[tmp_1.index, "bag_weight_value"] = tmp_1["bag_weight_value"]
    tmp_df.loc[tmp_2.index, "bag_weight_value"] = tmp_2["bag_weight_value"]
    tmp_df.loc[tmp_3.index, "bag_weight_value"] = tmp_3["bag_weight_value"]

    output_df["bag_weight"] = tmp_df["bag_weight_value"]
    output_df = output_df.rename(columns={"bag_weight": "bag_weight_g"})
    output_df["bag_weight_log_g"] = np.log1p(output_df["bag_weight_g"])

    return output_df


def uniform_harvest_year(input_df: pd.DataFrame):
    output_df = input_df.copy()
    key_year = [
        "2012",
        "2014",
        "2013",
        "2015",
        "2016",
        "2017",
        "2013/2014",
        "2015/2016",
        "2011",
        "2017 / 2018",
        "2014/2015",
        "2009/2010",
        "2010",
        "2010-2011",
        "2016 / 2017",
        "4T/10",
        "Mayo a Julio",
        "March 2010",
        "4T/2010",
        "2009-2010",
        "08/09 crop",
        "2011/2012",
        "January 2011",
        "Abril - Julio",
        "May-August",
        "2016/2017",
        "4T72010",
        "Sept 2009 - April 2010",
        "August to December",
        "23 July 2010",
        "2009 - 2010",
        "4t/2010",
        "1t/2011",
        "2009 / 2010",
        "Spring 2011 in Colombia.",
        "TEST",
        "mmm",
        "4t/2011",
        "47/2010",
        "December 2009-March 2010",
        "3T/2011",
        "Abril - Julio /2011",
        "1T/2011",
        "2018",
        "January Through April",
        "Fall 2009",
    ]

    val_year = [
        2012,
        2014,
        2013,
        2015,
        2016,
        2017,
        2013,
        2015,
        2011,
        2017,
        2014,
        2009,
        2010,
        2010,
        2016,
        2010,
        np.nan,
        2010,
        2010,
        2009,
        np.nan,
        2011,
        2011,
        np.nan,
        np.nan,
        2016,
        2010,
        2009,
        np.nan,
        2010,
        2009,
        2010,
        2011,
        2009,
        2011,
        np.nan,
        np.nan,
        2011,
        2010,
        2009,
        2011,
        2011,
        2011,
        2018,
        np.nan,
        2009,
    ]

    harvest_year_dict = dict(zip(key_year, val_year))
    output_df["harvest_year"] = output_df["harvest_year"].map(harvest_year_dict)

    return output_df


def add_grading_date_features(input_df: pd.DataFrame):
    output_df = input_df.copy()
    tmp_df = pd.to_datetime(input_df["grading_date"])
    output_df["grading_date_julian"] = tmp_df.map(pd.Timestamp.to_julian_date)
    output_df["grading_date_year"] = tmp_df.dt.year
    output_df["grading_date_month"] = tmp_df.dt.month
    output_df["grading_date_day"] = tmp_df.dt.day
    date_raw = (
        output_df["grading_date_year"].astype(str)
        + output_df["grading_date_month"].astype(str)
        + output_df["grading_date_day"].astype(str)
    )
    output_df["grading_date_raw"] = date_raw.astype(int)

    return output_df


def add_expiration_features(input_df: pd.DataFrame):
    output_df = input_df.copy()
    tmp_df = pd.to_datetime(input_df["expiration"])
    output_df["expiration_julian"] = tmp_df.map(pd.Timestamp.to_julian_date)
    output_df["expiration_year"] = tmp_df.dt.year
    output_df["expiration_month"] = tmp_df.dt.month
    output_df["expiration_day"] = tmp_df.dt.day
    date_raw = (
        output_df["expiration_year"].astype(str)
        + output_df["expiration_month"].astype(str)
        + output_df["expiration_day"].astype(str)
    )
    output_df["expiration_raw"] = date_raw.astype(int)

    return output_df


def add_defects_features(input_df: pd.DataFrame):
    output_df = input_df.copy()
    output_df["category_sum_defects"] = (
        output_df["category_one_defects"] + output_df["category_two_defects"]
    )
    output_df["category_one_and_two_defects_raw"] = output_df[
        "category_one_defects"
    ].astype(str) + output_df["category_two_defects"].astype(str)
    output_df["is_specialty"] = (output_df["category_one_defects"] == 0) & (
        output_df["category_two_defects"] <= 5
    )
    output_df["is_specialty"] = output_df["is_specialty"].astype(int)

    return output_df


def add_cross_num_features(input_df: pd.DataFrame):
    output_df = input_df.copy()
    output_df["yield"] = output_df["numberof_bags"] * output_df["bag_weight_g"]
    output_df["log_yield"] = output_df["numberof_bags"] * output_df["bag_weight_log_g"]

    return output_df


def to_preprocess(
    config: DictConfig, train_df: pd.DataFrame, test_df: pd.DataFrame, logger
):
    input_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    preprocess_config = config["feature"]["processes"]
    processes = preprocess_config["to_preprocess"]

    logger.info(f"All processing method : {processes}")

    output_df = input_df.copy()

    for func in tqdm(processes):
        logger.info(f"Now processing method : {func}")
        func = globals().get(func)
        output_df = func(output_df)

    _train_df = output_df.iloc[: len(train_df)]
    _test_df = output_df.iloc[len(train_df) :].reset_index(drop=True)

    return _train_df, _test_df
