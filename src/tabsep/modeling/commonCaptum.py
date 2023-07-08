import os

import numpy as np
import torch
from captum.attr import (
    GuidedBackprop,
    GuidedGradCam,
    InputXGradient,
    IntegratedGradients,
    ShapleyValueSampling,
)

torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


CORES_AVAILABLE = len(os.sched_getaffinity(0))


def captum_runner(trained_model: torch.nn.Module, X):
    X = X.to("cuda")
    trained_model = trained_model.to("cuda")

    model_name = trained_model.__class__.__name__
    trained_model.eval()

    X.requires_grad = True

    attributor = IntegratedGradients(trained_model)
    attributions = attributor.attribute(X, target=0)

    os.makedirs(f"cache/captum/{model_name}", exist_ok=True)
    torch.save(attributions, f"cache/captum/{model_name}/attributions.pt")
    torch.save(X, f"cache/captum/{model_name}/X.pt")
