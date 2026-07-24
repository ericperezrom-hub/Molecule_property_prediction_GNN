import torch
import pytest
from training.components import build_training_components


def make_config(optimizer_type="adam", criterion_type="mse"):
    return {
        "model": {"model_type": "gcn", "num_layers": 1, "hidden_dim": 4},
        "training": {"lr": 0.01, "optimizer_type": optimizer_type, "criterion_type": criterion_type},
    }


def test_build_training_components_returns_expected():
    config = make_config(optimizer_type="adam", criterion_type="mse")
    device = torch.device("cpu")
    num_node_features = 3

    model, optimizer, criterion = build_training_components(config, device, num_node_features)

    assert model is not None
    assert optimizer is not None
    assert criterion is not None
    assert isinstance(criterion, torch.nn.MSELoss)


def test_build_training_components_unknown_optimizer():
    config = make_config(optimizer_type="unknown", criterion_type="mse")
    device = torch.device("cpu")
    num_node_features = 3

    with pytest.raises(ValueError, match="Unknown optimizer_type"):
        build_training_components(config, device, num_node_features)


def test_build_training_components_unknown_criterion():
    config = make_config(optimizer_type="adam", criterion_type="unknown")
    device = torch.device("cpu")
    num_node_features = 3

    with pytest.raises(ValueError, match="Unknown criterion_type"):
        build_training_components(config, device, num_node_features)
