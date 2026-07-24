import torch
import pytest
from models.factory import build_model


def make_config(model_type="gcn", num_layers=1, hidden_dim=4):
    return {
        "model": {
            "model_type": model_type,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
        }
    }


def test_build_model_returns_gcn():
    config = make_config(model_type="gcn", num_layers=1, hidden_dim=4)
    device = torch.device("cpu")

    model = build_model(config, device, num_node_features=3)

    assert model is not None
    assert next(model.parameters()).device == device


def test_build_model_unknown_type():
    config = make_config(model_type="unknown", num_layers=1, hidden_dim=4)
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="Unknown model_type"):
        build_model(config, device, num_node_features=3)
