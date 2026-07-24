import torch
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from training.train_utils import setup_environment, prepare_data, train_and_validate


def make_config():
    return {
        "data": {"seed": 42, "target_idx": 0, "batch_size": 1},
        "logging": {"config_debug": False, "gpu_debug": False},
        "training": {"n_epochs": 3, "min_delta": 0.01, "patience": 1},
    }


def test_setup_environment_patches_config_and_device(monkeypatch):
    config = make_config()

    monkeypatch.setattr("training.train_utils.validate_config", lambda cfg: None)
    monkeypatch.setattr("training.train_utils.set_seed", lambda seed: None)
    monkeypatch.setattr("training.train_utils.get_git_commit_hash", lambda: "test-hash")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config_out, device = setup_environment(config)

    assert config_out["git_commit"] == "test-hash"
    assert device.type == "cpu"
    assert config_out is not config


def test_prepare_data_uses_dataloader_and_stats(monkeypatch):
    config = make_config()
    device = torch.device("cpu")
    dummy_loader = object()
    train_data = object()
    num_node_features = 5

    monkeypatch.setattr("training.train_utils.get_dataloader", lambda cfg: (dummy_loader, dummy_loader, dummy_loader, train_data, object(), num_node_features))
    monkeypatch.setattr("training.train_utils.get_target_stats", lambda train_data, target_idx: (torch.tensor(1.0), torch.tensor(2.0)))

    train_loader, val_loader, mean, std, out_num_node_features = prepare_data(config, device)

    assert train_loader is dummy_loader
    assert val_loader is dummy_loader
    assert torch.equal(mean, torch.tensor(1.0))
    assert torch.equal(std, torch.tensor(2.0))
    assert out_num_node_features == 5


def test_train_and_validate_saves_best_model(monkeypatch, tmp_path):
    model = MagicMock()
    optimizer = MagicMock()
    criterion = MagicMock()
    device = torch.device("cpu")
    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)
    run = SimpleNamespace(id="123")
    config = {"training": {"n_epochs": 3, "min_delta": 0.01, "patience": 1}}

    train_losses = [1.0, 1.0, 1.0]
    val_losses = [0.5, 0.4, 0.41]

    def fake_train(*args, **kwargs):
        return train_losses.pop(0)

    def fake_validate(*args, **kwargs):
        return val_losses.pop(0)

    monkeypatch.setattr("training.train_utils.train_one_epoch", fake_train)
    monkeypatch.setattr("training.train_utils.validate", fake_validate)
    monkeypatch.setattr("training.train_utils.wandb", MagicMock())
    saved = {}

    def fake_save(state, path):
        saved[path] = state

    monkeypatch.setattr(torch, "save", fake_save)

    best_val_loss = train_and_validate(
        model,
        None,
        None,
        optimizer,
        criterion,
        device,
        mean,
        std,
        target_idx=0,
        config=config,
        run=run,
    )

    assert best_val_loss == pytest.approx(0.4)
    assert any(path.endswith("best_model_123.pt") for path in saved)
    assert "model_state_dict" in saved[next(iter(saved))]
