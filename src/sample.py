import hydra
import os
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path


@hydra.main(config_path="../config", config_name="exp_001.yaml")
def main(cfg: DictConfig):
    print(cfg["params"])
    print(f"Current working directory: {os.getcwd()}")
    print(f"Orig working directory : {hydra.utils.get_original_cwd()}")
    print("------Train data loading------")
    dir_path = Path(hydra.utils.get_original_cwd())
    print(dir_path.joinpath("..", "input/train.csv"))
    train = pd.read_csv(dir_path.joinpath("..", "input/train.csv"))
    print(train.columns)


if __name__ == "__main__":
    main()
