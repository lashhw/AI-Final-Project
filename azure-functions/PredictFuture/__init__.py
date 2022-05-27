import logging
import azure.functions as func
from azure.data.tables import TableServiceClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import configparser
import json
from .ParkingModel import ParkingModel
from .utility import lstm_predict, prophet_predict

config = configparser.ConfigParser()
config.read('config.ini')
connection_string = config['CONNECTION']['ConnectionString']

table_service = TableServiceClient.from_connection_string(connection_string)
table_client = table_service.get_table_client('parking')

ids_train = np.load('./PredictFuture/ids_train.npy')
scaler_min = np.load('./PredictFuture/scaler_min.npy')
scaler_max = np.load('./PredictFuture/scaler_max.npy')

parking_model = ParkingModel(250, 96)
parking_model.load_state_dict(torch.load('./PredictFuture/parking_model.pt', map_location=torch.device('cpu')))
parking_model.eval()


def main(req: func.HttpRequest) -> func.HttpResponse:
    id_str = req.params.get('id')
    if not id_str:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass the ID of a specific parking lot to predict future trend.",
            status_code=200
        )

    id_name = '_' + id_str
    date_start = (datetime.utcnow()-timedelta(days=3)).strftime('%Y-%m-%d')

    entities = table_client.query_entities(f"PartitionKey ge '{date_start}'", select=['RowKey', id_name])
    entities_list = list(entities)

    times = np.array([datetime.strptime(x['RowKey'], '%Y-%m-%d %H:%M') for x in entities_list])
    data = np.array([int(x[id_name]) for x in entities_list])

    id = int(id_str)
    df = pd.DataFrame(data, index=times, columns=[id])

    lstm_pred, lstm_pred_times = lstm_predict(df, ids_train, scaler_min, scaler_max, parking_model)
    lstm_pred_times_str = [x.strftime('%Y-%m-%d %H:%M') for x in lstm_pred_times]
    lstm_pred_pairs = dict(zip(lstm_pred_times_str, lstm_pred.squeeze().tolist()))

    prophet_pred, prophet_pred_times = prophet_predict(id)
    prophet_pred_times_str = [x.strftime('%Y-%m-%d %H:%M') for x in prophet_pred_times]
    prophet_pred_pairs = dict(zip(prophet_pred_times_str, prophet_pred))

    pred_str = json.dumps({'lstm': lstm_pred_pairs, 'prophet': prophet_pred_pairs})

    return func.HttpResponse(
        pred_str,
        mimetype="application/json",
        status_code=200
    )
