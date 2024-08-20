import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBiLSTM(nn.Module):
    def __init__(self):
        super(ConvBiLSTM, self).__init__()

        # Convolutional layer
        self.conv1 = nn.Conv1d(8, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(256, 1)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)

        # Pass through BiLSTM
        x, _ = self.bilstm(x)

        # Take the output of the last time step
        x = self.fc(x[:, -1, :])

        return x


if __name__ == "__main__":
    fake_data = torch.randn(10, 24, 8)
    x = fake_data.permute(0, 2, 1)

    model = ConvBiLSTM()
    output = model(x)
    print(output.shape)  # Dự kiến là [batch_size, output_size]
