import torch
import torch.nn as nn
import torch.nn.functional as F


class Cnnlstm(nn.Module):
    def __init__(self):
        super(Cnnlstm, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)

        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(num_features=64)

        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=5, padding=2
        )
        self.bn3 = nn.BatchNorm1d(num_features=128)

        self.conv4 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=5, padding=2
        )
        self.bn4 = nn.BatchNorm1d(num_features=128)

        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.permute(0, 2, 1)

        x, (hn, cn) = self.lstm(x)
        x = self.global_avg_pool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    fake_data = torch.randn(10, 24, 8)
    x = fake_data.permute(0, 2, 1)

    model = Cnnlstm()

    output = model(x)
    print(output.shape)
