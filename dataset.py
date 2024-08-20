import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


def create_time_series_dataloader(X, y, batch_size=32, shuffle=True, num_workers=0):
    dataset = TimeSeriesDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    data_train = np.load("data\dataset_Taiwan\Banqiao_train.npz")
    data_test = np.load("data\dataset_Taiwan\Banqiao_test.npz")

    train_loader = create_time_series_dataloader(
        data_train["samples"], data_train["labels"], batch_size=32
    )
    test_loader = create_time_series_dataloader(
        data_test["samples"], data_test["labels"], batch_size=32, shuffle=False
    )

    print("Testing train_loader:")
    for i, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")
        # print(f"  First input sample: {inputs[0]}")
        # print(f"  First target sample: {targets[0]}")
        if i == 2:  # Chỉ test 3 batch đầu tiên
            break

    # Test test_loader
    print("\nTesting test_loader:")
    for i, (inputs, targets) in enumerate(test_loader):
        print(f"Batch {i+1}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")
        # print(f"  First input sample: {inputs[0]}")
        # print(f"  First target sample: {targets[0]}")
        if i == 2:  # Chỉ test 3 batch đầu tiên
            break
