import torch
import torch.nn as nn
from torchinfo import summary
from thop import profile


class RepeatVector(nn.Module):
    def __init__(self, n_repeats):
        super(RepeatVector, self).__init__()
        self.n_repeats = n_repeats

    def forward(self, x):
        return x.unsqueeze(1).repeat(1, self.n_repeats, 1)


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len):
        super(BiLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # Bi-LSTM 1
        self.bilstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # Bi-LSTM 2
        self.bilstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # Repeat Vector
        self.repeat_vector = RepeatVector(seq_len)

        # Bi-LSTM 3 (sau RepeatVector)
        self.bilstm3 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # Bi-LSTM 4
        self.bilstm4 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Bi-LSTM 1
        out, (h, c) = self.bilstm1(x)

        # Bi-LSTM 2
        out, _ = self.bilstm2(out)

        # Lấy đầu ra từ bước thời gian cuối cùng và lặp lại
        last_step_output = out[:, -1, :]
        repeated_output = self.repeat_vector(last_step_output)

        # Bi-LSTM 3
        out, _ = self.bilstm3(repeated_output)

        # Bi-LSTM 4
        out, _ = self.bilstm4(out)

        # Fully connected layer
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    fake_data = torch.randn(10, 24, 8)
    # x = fake_data.permute(0, 2, 1)

    model = BiLSTMModel(input_size=8, hidden_size=16, seq_len=24)

    output = model(fake_data)
    print(output.shape)
    # summary(model, fake_data.shape)
    # In thông tin mô hình
    # print(model)
    # Tính FLOPs và số lượng tham số
    flops, params = profile(model, inputs=(fake_data,))

    # In FLOPs và số lượng tham số
    print(f"FLOPs: {flops}, Params: {params}")
