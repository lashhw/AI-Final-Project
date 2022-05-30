import torch
from torch import nn
import torch.nn.functional as F


class ParkingModel(nn.Module):
    def __init__(self, p_lot_len, window_len, predict_len):
        super().__init__()
        
        self.p_lot_len = p_lot_len
        self.window_len = window_len
        self.predict_len = predict_len

        self.rnn = nn.LSTM(1, 128, num_layers=1, batch_first=True)

        self.fc1 = nn.Linear(128+9+p_lot_len, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, predict_len)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, input_seq, input_extra):
        assert input_seq.ndim == 2 and input_seq.shape[1] == self.window_len

        input_seq = input_seq.unsqueeze(2)
        # (N, L, 1)

        x, _ = self.rnn(input_seq)
        # (N, L, 128)
        x = x[:, -1, :]
        x = self.bn1(x)
        # (N, 128)

        x = torch.cat((x, input_extra), dim=1)
        # (N, 128+9+p_lot_len)
        
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        # (N, 128)

        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        # (N, 128)

        x = self.fc3(x)
        # (N, predict_len)

        return x