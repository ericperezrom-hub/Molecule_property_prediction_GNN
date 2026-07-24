import pytest
from unittest.mock import patch

from dataloader import load_split_data, get_dataloader


def make_config(data_debug=False, size_debug=3, batch_size=1):
    return {
        "data": {
            "data_debug": data_debug,
            "size_debug": size_debug,
            "batch_size": batch_size,
        }
    }


def test_load_split_data_with_debug(monkeypatch):
    config = make_config(data_debug=True, size_debug=4)

    class DummySubset:
        def __init__(self, data, num_node_features):
            self._data = data
            self.num_node_features = num_node_features

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return DummySubset(self._data[idx], self.num_node_features)
            return self._data[idx]

    class DummyDataset:
        def __init__(self):
            self.num_node_features = 5
            self._data = [1, 2, 3, 4]

        def shuffle(self):
            return self

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return DummySubset(self._data[idx], self.num_node_features)
            return self._data[idx]

    monkeypatch.setattr("dataloader.pyg.QM9", lambda path: DummyDataset())

    train_data, val_data, test_data, num_node_features = load_split_data(config)

    assert len(train_data) == 3
    assert len(val_data) == 0
    assert len(test_data) == 1
    assert num_node_features == 5


def test_get_dataloader(monkeypatch):
    config = make_config(data_debug=True, size_debug=4, batch_size=2)

    class DummySubset:
        def __init__(self, data, num_node_features):
            self._data = data
            self.num_node_features = num_node_features

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return DummySubset(self._data[idx], self.num_node_features)
            return self._data[idx]

    class DummyDataset:
        def __init__(self):
            self.num_node_features = 5
            self._data = [1, 2, 3, 4]

        def shuffle(self):
            return self

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return DummySubset(self._data[idx], self.num_node_features)
            return self._data[idx]

    monkeypatch.setattr("dataloader.pyg.QM9", lambda path: DummyDataset())
    monkeypatch.setattr("dataloader.DataLoader", lambda data, batch_size, shuffle: (data, batch_size, shuffle))

    train_loader, val_loader, test_loader, train_data, test_data, num_node_features = get_dataloader(config)

    assert train_loader[1] == 2
    assert val_loader[2] is False
    assert test_loader[2] is False
    assert len(train_data) == 3
    assert list(train_data._data) == [1, 2, 3]
    assert len(test_data) == 1
    assert list(test_data._data) == [4]
    assert num_node_features == 5
