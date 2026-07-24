import os
import tempfile
import subprocess
import random
import torch
import numpy as np
import yaml
from unittest.mock import patch

from utils import (
    load_config,
    print_config,
    print_gpu_stats,
    validate_config,
    get_git_commit_hash,
    set_seed,
    REQUIRED_CONFIG,
)


def test_load_config(tmp_path):
    config_path = tmp_path / "config_test.yaml"
    config_content = {
        "data": {"batch_size": 32, "target_idx": 4, "seed": 123},
        "model": {"model_type": "gcn", "hidden_dim": 16, "num_layers": 2},
        "training": {"n_epochs": 1, "lr": 0.01, "patience": 1, "min_delta": 0.0, "optimizer_type": "adam", "criterion_type": "mse"},
        "logging": {"demo_project": "test", "gpu_debug": False, "config_debug": False},
    }
    config_path.write_text(yaml.safe_dump(config_content))

    loaded = load_config(str(config_path))

    assert loaded == config_content


def test_print_config(capsys):
    config = {
        "data": {"batch_size": 8, "target_idx": 0, "seed": 1},
        "other": "value",
    }

    print_config(config)
    captured = capsys.readouterr()

    assert "USED CONFIGURATION:" in captured.out
    assert "data:" in captured.out
    assert "batch_size" in captured.out
    assert "other: value" in captured.out


def test_print_gpu_stats_no_cuda(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    print_gpu_stats()
    captured = capsys.readouterr()

    assert "CUDA available: False" in captured.out


def test_validate_config_missing_key():
    config = {
        "data": {"batch_size": 32, "seed": 1},
        "model": {"model_type": "gcn", "hidden_dim": 8, "num_layers": 1},
        "training": {"n_epochs": 1, "lr": 0.01, "patience": 1, "min_delta": 0.0, "optimizer_type": "adam", "criterion_type": "mse"},
        "logging": {"demo_project": "test", "gpu_debug": False, "config_debug": False},
    }

    try:
        validate_config(config)
        assert False, "Expected ValueError for missing target_idx"
    except ValueError as exc:
        assert "data.target_idx" in str(exc)


def test_get_git_commit_hash_success(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: b"abcdef123456\n")

    commit_hash = get_git_commit_hash()

    assert commit_hash == "abcdef123456"


def test_get_git_commit_hash_failure(monkeypatch):
    def raise_error(*args, **kwargs):
        raise subprocess.CalledProcessError(1, cmd=args[0])

    monkeypatch.setattr(subprocess, "check_output", raise_error)

    commit_hash = get_git_commit_hash()

    assert commit_hash == "unknown"


def test_set_seed_reproducible():
    set_seed(1234)

    values = [random.random() for _ in range(3)]
    np_values = np.random.rand(3).tolist()
    torch_values = torch.rand(3).tolist()

    set_seed(1234)

    assert values == [random.random() for _ in range(3)]
    assert np_values == np.random.rand(3).tolist()
    assert torch_values == torch.rand(3).tolist()
