import torch
from torch import nn
import torch.nn.functional as F


class ParkingModel(nn.Module):
    def __init__(self, p_lot_len, predict_len):
        super().__init__()
        
        self.p_lot_len = p_lot_len
        self.predict_len = predict_len

        self.rnn = nn.LSTM(1, 128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128+9+p_lot_len, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, predict_len)

    def forward(self, input_seq, input_extra, h_in=None, feed_last_only=False):
        batch_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        input_seq = input_seq.unsqueeze(2)
        assert input_seq.shape == (batch_size, seq_len, 1)

        if h_in is None:
            out, h_out = self.rnn(input_seq)
        else:
            out, h_out = self.rnn(input_seq, h_in)
        assert out.shape == (batch_size, seq_len, 128)

        if feed_last_only:
            x = out[:, -1, :]
            assert x.shape == (batch_size, 128)

            assert input_extra.shape == (batch_size, 9+self.p_lot_len)

            x = torch.cat((x, input_extra), dim=1)
            assert x.shape == (batch_size, 128+9+self.p_lot_len)

            x = F.relu(self.fc1(x))
            assert x.shape == (batch_size, 128)

            x = F.relu(self.fc2(x))
            assert x.shape == (batch_size, 128)

            x = self.fc3(x)
            assert x.shape == (batch_size, self.predict_len)
        else:
            assert input_extra.shape == (batch_size, seq_len, 9+self.p_lot_len)

            x = torch.cat((out, input_extra), dim=2)
            assert x.shape == (batch_size, seq_len, 128+9+self.p_lot_len)

            x = F.relu(self.fc1(x))
            assert x.shape == (batch_size, seq_len, 128)

            x = F.relu(self.fc2(x))
            assert x.shape == (batch_size, seq_len, 128)

            x = self.fc3(x)
            assert x.shape == (batch_size, seq_len, self.predict_len)

        return x, h_out
