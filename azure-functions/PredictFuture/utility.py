from datetime import timedelta
import numpy as np
import torch
import torch.nn.functional as F
import json
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


def prophet_predict(id):
	from prophet.serialize import model_from_json
	with open(f'./PredictFuture/prophet_models/{id}.json') as f:
		m = model_from_json(json.load(f))
	
	start_ts = pd.Timestamp.now().ceil('15min')
	future_dt = pd.date_range(start=start_ts, periods=96, freq='15min')
	future_df = future_dt.to_frame(name='ds').reset_index(drop=True)

	forecast = m.predict(future_df)
	return forecast['yhat'], future_dt
