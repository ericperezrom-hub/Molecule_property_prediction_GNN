import torch
import pytest
from torch import nn
from types import SimpleNamespace
from training.loop import get_target_stats, train_one_epoch, validate


class DummyBatch:
    def __init__(self, y, num_graphs):
        self.y = y
        self.num_graphs = num_graphs
        self.device = y.device

    def to(self, device):
        self.y = self.y.to(device)
        return self


class DummyLoader:
    class DatasetWrapper:
        def __init__(self, dataset_len):
            self._len = dataset_len

        def __len__(self):
            return self._len

    def __init__(self, batches, dataset_len):
        self._batches = batches
        self.dataset = DummyLoader.DatasetWrapper(dataset_len)

    def __iter__(self):
        return iter(self._batches)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, data):
        out = torch.ones((data.num_graphs, 1), dtype=torch.float, requires_grad=True)
        return out


def test_get_target_stats():
    class TrainData:
        def __init__(self, y):
            self.y = y

    data = TrainData(y=torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]))

    mean, std = get_target_stats(data, target_idx=2)

    assert pytest.approx(mean.item(), rel=1e-6) == 4.5
    assert pytest.approx(std.item(), rel=1e-6) == pytest.approx(2.1213203435596424)


def test_train_one_epoch_and_validate():
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()
    device = torch.device("cpu")

    y = torch.tensor([[1.0, 2.0, 3.0]])
    batch = DummyBatch(y=y, num_graphs=1)
    loader = DummyLoader(batches=[batch], dataset_len=1)

    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)
    target_idx = 2

    train_loss = train_one_epoch(model, loader, optimizer, criterion, device, mean, std, target_idx)
    val_loss = validate(model, loader, criterion, device, mean, std, target_idx)

    assert train_loss >= 0.0
    assert val_loss >= 0.0
    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)


def test_validate_handles_multiple_batches():
    model = DummyModel()
    criterion = torch.nn.MSELoss()
    device = torch.device("cpu")

    y1 = torch.tensor([[0.0, 0.0, 1.0]])
    y2 = torch.tensor([[0.0, 0.0, 2.0]])
    batch1 = DummyBatch(y=y1, num_graphs=1)
    batch2 = DummyBatch(y=y2, num_graphs=1)
    loader = DummyLoader(batches=[batch1, batch2], dataset_len=2)

    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)
    target_idx = 2

    loss = validate(model, loader, criterion, device, mean, std, target_idx)

    assert loss >= 0.0
    assert loss == pytest.approx(loss)
