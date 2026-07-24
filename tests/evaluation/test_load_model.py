import torch
from unittest.mock import MagicMock
from evaluation.load_model import load_model_from_checkpoint


def test_load_model_from_checkpoint(monkeypatch):
    fake_checkpoint = {
        "config": {"data": {"target_idx": 2}},
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "mean": torch.tensor(0.5),
        "std": torch.tensor(2.0),
    }

    fake_model = MagicMock()
    fake_model.load_state_dict.return_value = None
    fake_model.eval.return_value = None

    monkeypatch.setattr("evaluation.load_model.torch.load", lambda path, map_location=None: fake_checkpoint)
    monkeypatch.setattr("evaluation.load_model.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("evaluation.load_model.get_dataloader", lambda cfg: (None, None, "test_loader", None, "test_data", 3))
    monkeypatch.setattr("evaluation.load_model.build_model", lambda cfg, device, num_node_features: fake_model)

    bundle = load_model_from_checkpoint("dummy_path")

    assert bundle["model"] is fake_model
    assert bundle["mean"] == torch.tensor(0.5)
    assert bundle["std"] == torch.tensor(2.0)
    assert bundle["config"]["data"]["target_idx"] == 2
    assert bundle["test_loader"] == "test_loader"
    assert bundle["test_data"] == "test_data"
