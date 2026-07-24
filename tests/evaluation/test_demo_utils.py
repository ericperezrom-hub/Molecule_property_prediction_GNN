import pytest
from unittest.mock import MagicMock

from evaluation.demo_utils import predict_all, select_examples, mol_to_image, log_demo_table


class DummyModel:
    def eval(self):
        return self

    def __call__(self, batch):
        return batch.predictions


class DummyBatch:
    def __init__(self, y, batch, predictions):
        self.y = y
        self.batch = batch
        self.num_graphs = batch.max().item() + 1
        self.predictions = predictions

    def to(self, device):
        return self


class DummySample:
    def __init__(self, smiles):
        self.smiles = smiles


def test_predict_all_returns_results():
    model = DummyModel()
    y = [[1.0, 2.0, 3.0]]
    batch = DummyBatch(
        y=__import__("torch").tensor([[1.0, 2.0, 3.0]]),
        batch=__import__("torch").tensor([0]),
        predictions=__import__("torch").tensor([[1.5]]),
    )
    results = predict_all(model, [batch], "cpu", __import__("torch").tensor(0.0), __import__("torch").tensor(1.0), target_idx=2)

    assert len(results) == 1
    assert results[0]["index"] == 0
    assert results[0]["prediction"] == pytest.approx(1.5)
    assert results[0]["ground_truth"] == pytest.approx(3.0)
    assert results[0]["error"] == pytest.approx(1.5)


def test_select_examples_modes():
    results = [
        {"error": 1.0},
        {"error": 3.0},
        {"error": 2.0},
    ]

    assert select_examples(results, 2, mode="best") == [{"error": 1.0}, {"error": 2.0}]
    assert select_examples(results, 2, mode="worst") == [{"error": 3.0}, {"error": 2.0}]
    assert len(select_examples(results, 2, mode="random")) == 2

    with pytest.raises(ValueError, match="Unknown mode"):
        select_examples(results, 1, mode="invalid")


def test_log_demo_table_calls_wandb(monkeypatch):
    fake_wandb = MagicMock()
    fake_table = MagicMock()
    fake_wandb.Table.return_value = fake_table
    fake_wandb.Image.return_value = "image-object"
    fake_wandb.init.return_value = None

    monkeypatch.setattr("evaluation.demo_utils.wandb", fake_wandb)
    monkeypatch.setattr("evaluation.demo_utils.mol_to_image", lambda smiles: "image-data")

    examples = [{"index": 0, "prediction": 1.0, "ground_truth": 2.0, "error": 1.0}]
    test_data = [DummySample(smiles="CC")]
    config = {"logging": {"demo_project": "demo"}}

    log_demo_table(examples, test_data, config)

    fake_wandb.Table.assert_called_once()
    fake_table.add_data.assert_called_once_with(
        "image-object",
        "CC",
        1.0,
        2.0,
        1.0,
    )
    fake_wandb.log.assert_called_once()
    fake_wandb.finish.assert_called_once()
