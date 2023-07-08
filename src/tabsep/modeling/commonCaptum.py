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


def captum_runner(trained_model: torch.nn.Module, X, batch_size=2):
    X = X.to("cuda")
    trained_model = trained_model.to("cuda")

    model_name = trained_model.__class__.__name__
    trained_model.eval()

    X.requires_grad = True

    attributor = IntegratedGradients(trained_model)
    attributions = list()

    for batch_idx in tqdm(range(0, X.shape[0], batch_size)):
        end_idx = min(batch_idx + batch_size, X.shape[0])
        attributions.append(
            attributor.attribute(X[batch_idx:end_idx], target=0).to("cpu")
        )

    attributions = torch.concat(attributions, dim=0)

    assert attributions.shape == X.shape

    os.makedirs(f"cache/captum/{model_name}", exist_ok=True)
    torch.save(attributions, f"cache/captum/{model_name}/attributions.pt")
    torch.save(X, f"cache/captum/{model_name}/X.pt")
