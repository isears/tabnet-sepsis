import datetime
import os
from dataclasses import dataclass


@dataclass
class Config:
    # Timesteps in the mimicts database
    timestep_seconds: int
    # Number of timesteps before sepsis onset
    prediction_timesteps: int
    # Path to TST model to use in results generation
    model_path: str
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
    model_path="/gpfs/home/isears1/Repos/tabnet-sepsis/cache/models/singleTst_2022-08-30_19:55:07",
    cores_available=len(os.sched_getaffinity(0)),
    timestep=datetime.timedelta(seconds=timestep_seconds),
)
