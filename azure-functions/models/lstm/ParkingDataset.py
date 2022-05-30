import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F


class ParkingDataset(Dataset):
    def __init__(self, sequence, is_valid, weekdays, seconds,
                 window_len, predict_len):
        self.sequence = torch.tensor(sequence).to(torch.float)
        self.is_valid = torch.tensor(is_valid).to(torch.float)
        self.p_lot_len, self.seq_len = self.sequence.shape

        weekdays_onehot = F.one_hot(torch.tensor(weekdays), num_classes=7)
        # assert weekdays_onehot.shape == (self.seq_len, 7)

        seconds_in_day = 24 * 60 * 60
        seconds_rad = 2 * np.pi * (seconds/seconds_in_day)
        seconds_cos = torch.tensor(np.cos(seconds_rad)).unsqueeze(1)
        seconds_sin = torch.tensor(np.sin(seconds_rad)).unsqueeze(1)
        # assert seconds_cos.shape == (self.seq_len, 1)
        # assert seconds_sin.shape == (self.seq_len, 1)

        self.timeinfo = torch.cat([weekdays_onehot, seconds_cos, seconds_sin], dim=1) \
                             .to(torch.float)
        # assert self.timeinfo.shape == (self.seq_len, 9)

        self.window_len = window_len
        self.predict_len = predict_len


    def __len__(self):
        return self.p_lot_len * (self.seq_len-self.window_len-self.predict_len+1)


    def __getitem__(self, idx):
        p_lot_idx     = idx // (self.seq_len-self.window_len-self.predict_len+1)
        seq_start_idx = idx %  (self.seq_len-self.window_len-self.predict_len+1)

        x_seq = self.sequence[p_lot_idx, seq_start_idx:
                              seq_start_idx+self.window_len]
        # assert x_seq.shape == (self.window_len,)

        x_timeinfo = self.timeinfo[seq_start_idx+self.window_len-1]
        # assert x_timeinfo.shape == (9,)

        x_p_lot = F.one_hot(torch.tensor(p_lot_idx), num_classes=self.p_lot_len)
        # assert x_p_lot.shape == (self.p_lot_len,)

        x_extra = torch.cat([x_timeinfo, x_p_lot], dim=0)
        # assert x_extra.shape == (9+self.p_lot_len,)

        y = self.sequence[p_lot_idx, seq_start_idx+self.window_len:
                          seq_start_idx+self.window_len+self.predict_len]
        y_valid = self.is_valid[p_lot_idx, seq_start_idx+self.window_len:
                                seq_start_idx+self.window_len+self.predict_len]
        # assert y.shape == (self.predict_len,)
        # assert y_valid.shape == (self.predict_len,)

        return x_seq, x_extra, y, y_valid