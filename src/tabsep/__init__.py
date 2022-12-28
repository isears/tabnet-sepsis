import datetime
import os
from dataclasses import dataclass

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


@dataclass
class Config:
    # Timesteps in the mimicts database
    timestep_seconds: int
    # Number of timesteps before sepsis onset
    prediction_timesteps: int
    # Path to TST model to use in results generation
    tst_path: str
    lr_path: str
    cores_available: int
    timestep: datetime.timedelta


with open("mimicts/readme.txt", "r") as f:
    mimicts_config = f.readlines()

    for line in mimicts_config:
        if "timestep" in line:
            timestep_seconds = int(line.split("=")[-1])


config = Config(
    timestep_seconds=timestep_seconds,  # TODO: deprecated infavor of timestep
    prediction_timesteps=1,
    tst_path="cache/models/singleTst",
    lr_path="cache/models/singleLr",
    cores_available=len(os.sched_getaffinity(0)),
    timestep=datetime.timedelta(seconds=timestep_seconds),
)
