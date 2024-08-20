import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(self, d_model):
        super(InceptionBlock, self).__init__()
        dim = d_model // 4
        self.conv1 = nn.Conv1d(d_model, dim, kernel_size=1)

        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

        self.conv3 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)

        self.conv4 = nn.Conv1d(dim, dim, kernel_size=7, padding=3)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv1d(d_model, dim, kernel_size=1)
        self.bnv1x1 = nn.BatchNorm1d(num_features=dim)

        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        branch1 = self.conv1x1(self.maxpool(x))

        branch2 = self.conv1(x)
        branch2_1 = self.conv2(branch2)
        branch2_2 = self.conv3(branch2)
        branch2_3 = self.conv4(branch2)

        outputs = torch.cat([branch1, branch2_1, branch2_2, branch2_3], dim=1)
        outputs = F.relu(self.bn(outputs))
        return outputs


class ConvInceptionBiLSTM(nn.Module):
    def __init__(self, k_model, d_model):
        super(ConvInceptionBiLSTM, self).__init__()

        self.conv1 = nn.Conv1d(k_model, d_model, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm1d(d_model)

        self.conv_shortcut1 = nn.Conv1d(d_model, d_model, kernel_size=1, padding=0)
        self.bn1_shortcut1 = nn.BatchNorm1d(d_model)

        self.inception1 = InceptionBlock(d_model)
        self.inception2 = InceptionBlock(d_model)
        self.inception3 = InceptionBlock(d_model)

        self.conv_shortcut2 = nn.Conv1d(d_model, d_model, kernel_size=1, padding=0)
        self.bn1_shortcut2 = nn.BatchNorm1d(d_model)

        self.inception4 = InceptionBlock(d_model)
        self.inception5 = InceptionBlock(d_model)
        self.inception6 = InceptionBlock(d_model)

        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(d_model * 2, 1)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        shortcut = self.bn1_shortcut1(self.conv_shortcut1(x))

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = shortcut + x

        shortcut = F.relu(self.bn1_shortcut2(self.conv_shortcut2(x)))
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = shortcut + x

        x = x.permute(0, 2, 1)  # Đổi thứ tự để phù hợp với LSTM: BxLxD
        x, _ = self.bilstm(x)

        x = self.fc(x[:, -1, :])  # Lấy giá trị tại thời điểm cuối cùng

        return x


if __name__ == "__main__":
    fake_data = torch.randn(10, 24, 8)
    x = fake_data.permute(0, 2, 1)

    model = ConvInceptionBiLSTM(8, 128)

    output = model(x)
    print(output.shape)
