import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F


class ParkingDataset(Dataset):
    def __init__(self, sequence, is_valid, weekdays, seconds, predict_len):
        self.sequence = torch.tensor(sequence).to(torch.float)
        self.is_valid = torch.tensor(is_valid).to(torch.float)
        self.p_lot_len, self.seq_len = self.sequence.shape

        weekdays_onehot = F.one_hot(torch.tensor(weekdays), num_classes=7)
        assert weekdays_onehot.shape == (self.seq_len, 7)

        seconds_in_day = 24 * 60 * 60
        seconds_rad = 2 * np.pi * (seconds/seconds_in_day)
        seconds_cos = torch.tensor(np.cos(seconds_rad)).unsqueeze(1)
        seconds_sin = torch.tensor(np.sin(seconds_rad)).unsqueeze(1)
        assert seconds_cos.shape == (self.seq_len, 1)
        assert seconds_sin.shape == (self.seq_len, 1)

        self.timeinfo = torch.cat([weekdays_onehot, seconds_cos, seconds_sin], dim=1) \
                             .to(torch.float)
        assert self.timeinfo.shape == (self.seq_len, 9)

        self.predict_len = predict_len


    def __len__(self):
        return self.p_lot_len


    def __getitem__(self, idx):
        part_seq_len = self.seq_len - self.predict_len

        x_seq = self.sequence[idx, :-self.predict_len]
        assert x_seq.shape == (part_seq_len,)

        x_timeinfo = self.timeinfo[:-self.predict_len]
        assert x_timeinfo.shape == (part_seq_len, 9)

        x_p_lot = F.one_hot(torch.tensor(idx), num_classes=self.p_lot_len)
        x_p_lot = x_p_lot.unsqueeze(0)
        assert x_p_lot.shape == (1, self.p_lot_len)

        x_p_lot = x_p_lot.expand(part_seq_len, -1)
        assert x_p_lot.shape == (part_seq_len, self.p_lot_len)

        x_extra = torch.cat([x_timeinfo, x_p_lot], dim=1)
        assert x_extra.shape == (part_seq_len, 9+self.p_lot_len)

        y = torch.empty(part_seq_len, self.predict_len)
        y_valid = torch.empty(part_seq_len, self.predict_len)
        for i in range(self.predict_len):
            y[:, i] = self.sequence[idx, i+1:i+1+part_seq_len]
            y_valid[:, i] = self.is_valid[idx, i+1:i+1+part_seq_len]
        assert y.shape == (part_seq_len, self.predict_len)
        assert y_valid.shape == (part_seq_len, self.predict_len)

        return x_seq, x_extra, y, y_valid