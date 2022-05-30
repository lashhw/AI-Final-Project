import torch
from torch import nn
import torch.nn.functional as F


class ParkingModel(nn.Module):
    def __init__(self, p_lot_len, predict_len, dropout_p=0.0):
        super().__init__()
        
        self.p_lot_len = p_lot_len
        self.predict_len = predict_len

        self.rnn = nn.LSTM(1, 128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128+9+p_lot_len, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, predict_len)

        self.bn_fc1 = nn.BatchNorm1d(128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, input_seq, input_extra, feed_last_only=False):
        x = input_seq.unsqueeze(2)
        # (N, L, 1)
        x, _ = self.rnn(x)
        # (N, L, 128)

        if feed_last_only:
            x = x[:, -1, :].unsqueeze(1)
            # (N, 1, 128)
            input_extra = input_extra.unsqueeze(1)
            # (N, 1, 9+p_lot_len)
        
        x = torch.cat((x, input_extra), dim=2)
        # (N, L, 128+9+p_lot_len)

        x = F.relu(self.fc1(x))
        # (N, L, 128)
        x = self.bn_fc1(x.transpose(1, 2)).transpose(1, 2)
        # (N, L, 128)
        x = self.dropout(x)
        # (N, L, 128)

        x = F.relu(self.fc2(x))
        # (N, L, 128)
        x = self.bn_fc2(x.transpose(1, 2)).transpose(1, 2)
        # (N, L, 128)
        x = self.dropout(x)
        # (N, L, 128)

        x = self.fc3(x)
        # (N, L, predict_len)

        if feed_last_only:
            x = x.squeeze(1)
            # (N, predict_len)

        return x
