import logging
import azure.functions as func
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import configparser
import json
from azure.data.tables import TableServiceClient
from .ParkingModel import ParkingModel
from .utility import evaluate

config = configparser.ConfigParser()
config.read('config.ini')
connection_string = config['CONNECTION']['ConnectionString']

ids_train = np.load('./PredictFuture/ids_train.npy')
scaler_min = np.load('./PredictFuture/scaler_min.npy')
scaler_max = np.load('./PredictFuture/scaler_max.npy')

parking_model = ParkingModel(250, 96)
parking_model.load_state_dict(torch.load('./PredictFuture/parking_model.pt', map_location=torch.device('cpu')))
parking_model.eval()


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('HTTP trigger function processed a request.')

    id_str = req.params.get('id')
    if not id_str:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass the ID of a specific parking lot to predict future trend.",
            status_code=200
        )

    table_service = TableServiceClient.from_connection_string(connection_string)
    table_client = table_service.get_table_client('parking')
    logging.info('Connection Established.')

    id_name = '_' + id_str
    date_start = (datetime.now()-timedelta(days=7)).strftime('%Y-%m-%d')

    entities = table_client.query_entities(f"PartitionKey ge '{date_start}'", select=['RowKey', id_name])
    entities_list = list(entities)

    times = np.array([datetime.strptime(x['RowKey'], '%Y-%m-%d %H:%M') for x in entities_list])
    data = np.array([int(x[id_name]) for x in entities_list])

    df = pd.DataFrame(data, index=times, columns=[int(id_str)])

    prediction, prediction_times = evaluate(df, ids_train, scaler_min, scaler_max, parking_model)
    prediction_times_str = [x.strftime('%Y-%m-%d %H:%M') for x in prediction_times]
    prediction_pairs = list(zip(prediction_times_str, 
                                prediction.squeeze().tolist()))
    prediction_str = json.dumps(prediction_pairs)

    return func.HttpResponse(
        prediction_str,
        mimetype="application/json",
        status_code=200
    )
