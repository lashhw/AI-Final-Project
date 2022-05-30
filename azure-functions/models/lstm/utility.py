from datetime import timedelta
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd


def preprocess(df):
    # change invalid (AVAILABLECAR <= -9) data to NaN
    df = df.mask(df<=-9.0)

    # fill to ensure all value are valid
    df = df.ffill().bfill()

    rolling_df = df.rolling(window='60min').mean()
    rolling_df.index = rolling_df.index - timedelta(minutes=30)

    smooth_df = rolling_df.resample('15min', offset='7.5min').mean()
    smooth_df.index = smooth_df.index + timedelta(minutes=7.5)
    smooth_df = smooth_df.ffill().bfill()
    assert smooth_df.isna().to_numpy().any() == False

    return df, smooth_df


def transform(df):
    data = df.to_numpy().T
    ids = np.array(df.columns)
    times = list(df.index)
    weekdays = np.array([x.weekday() for x in times])
    seconds = np.array([(x-x.replace(hour=0, minute=0)).seconds for x in times])
    return data, ids, times, weekdays, seconds


def get_valid(df, no_change_threshold):
    same_as_prev = df.diff(1) == 0
    is_valid = np.ones_like(df, dtype=bool)

    for i in range(df.shape[1]):
        same_count = 0
        for j in range(df.shape[0]):
            if same_as_prev.iloc[j, i] == True:
                same_count += 1
                if same_count == 1:
                    same_start = j
                elif same_count == no_change_threshold:
                    is_valid[same_start:j+1, i] = False
                elif same_count > no_change_threshold:
                    is_valid[j, i] = False
            else:
                same_count = 0

    return is_valid.T


def get_indices(ids, ids_train):
    assert np.isin(ids, ids_train).all()

    train_sort_idx = ids_train.argsort()
    indices = train_sort_idx[np.searchsorted(ids_train, ids, sorter=train_sort_idx)]
    return indices


def scale(data, ids, ids_train, scaler_min, scaler_max):
    i = get_indices(ids, ids_train)

    std = (data.T-scaler_min[i]) / (scaler_max[i]-scaler_min[i])
    scaled = std * 2 - 1

    return scaled.T


def inverse_scale(scaled, ids, ids_train, scaler_min, scaler_max):
    i = get_indices(ids, ids_train)

    std = (scaled.T+1) / 2
    data = std * (scaler_max[i]-scaler_min[i]) + scaler_min[i]

    return data.T


def lstm_predict(df, ids_train, scaler_min, scaler_max, parking_model):
    df, smooth_df = preprocess(df)
    data, ids, times, weekdays, seconds = transform(smooth_df)

    indices = get_indices(ids, ids_train)
    eval_p_lot_len = data.shape[0]

    scaled = scale(data, ids, ids_train, scaler_min, scaler_max)
    scaled = torch.tensor(scaled, dtype=torch.float)
    
    weekday_onehot = F.one_hot(torch.tensor(weekdays[-1]), num_classes=7)
    seconds_in_day = 24 * 60 * 60
    second_rad = 2 * np.pi * (seconds[-1]/seconds_in_day)
    second_cos = torch.tensor(np.cos(second_rad)).unsqueeze(0)
    second_sin = torch.tensor(np.sin(second_rad)).unsqueeze(0)
    time_info = torch.cat([weekday_onehot, second_cos, second_sin]) \
                     .expand(eval_p_lot_len, -1)
    assert time_info.shape == (eval_p_lot_len, 9)

    p_lot_onehot = F.one_hot(torch.tensor(indices), 
                             num_classes=parking_model.p_lot_len)
    assert p_lot_onehot.shape == (eval_p_lot_len, parking_model.p_lot_len)

    extra = torch.cat([time_info, p_lot_onehot], dim=1).to(torch.float)
    assert extra.shape == (eval_p_lot_len, 9+parking_model.p_lot_len)

    out, _ = parking_model(scaled, extra, feed_last_only=True)
    prediction = inverse_scale(out.detach().numpy(), ids, ids_train, scaler_min, scaler_max)
    prediction_times = [times[-1] + timedelta(minutes=15*(i+1)) for i in range(parking_model.predict_len)]
    
    return prediction, prediction_times


def prophet_predict(id, ids_train, pred_length, prophet_model, pred_start=None):
    if pred_start is None:
        pred_start = pd.Timestamp.utcnow()

    id_idx = get_indices([id], ids_train)[0]
    pred_start = pred_start.ceil('15min')
    start_idx = pred_start.weekday()*96 + \
                (pred_start-pred_start.replace(hour=0, minute=0)).seconds//900

    tmp_pred = np.concatenate([prophet_model, prophet_model], axis=2)
    pred = tmp_pred[id_idx, :, start_idx:start_idx+pred_length].tolist()

    future_dt = pd.date_range(start=pred_start, periods=pred_length, freq='15min')
    return pred, future_dt
