import os

import numpy as np
import torch
from captum.attr import (
    DeepLift,
    FeaturePermutation,
    GuidedBackprop,
    GuidedGradCam,
    InputXGradient,
    IntegratedGradients,
    Saliency,
    ShapleyValueSampling,
)
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


CORES_AVAILABLE = len(os.sched_getaffinity(0))


def captum_runner(model_name: str, trained_model: torch.nn.Module, X, batch_size=128):
    X = X.to("cuda")
    trained_model = trained_model.to("cuda")

    trained_model.eval()

    X.requires_grad = True

    attributor = Saliency(trained_model)
    attributions = list()

    for batch_idx in tqdm(range(0, X.shape[0], batch_size)):
        end_idx = min(batch_idx + batch_size, X.shape[0])
        attributions.append(
            attributor.attribute(X[batch_idx:end_idx], target=0).detach().to("cpu")
        )

    attributions = torch.concat(attributions, dim=0)

    assert attributions.shape == X.shape

    return attributions
