from dataclasses import dataclass


@dataclass
class Config:
    # Timesteps in the mimicts database
    timestep_seconds: int
    # Number of timesteps before sepsis onset
    prediction_timesteps: int


with open("mimicts/readme.txt", "r") as f:
    mimicts_config = f.readlines()

    for line in mimicts_config:
        if "timestep" in line:
            timestep_seconds = int(line.split("=")[-1])


config = Config(timestep_seconds=timestep_seconds, prediction_timesteps=6)
