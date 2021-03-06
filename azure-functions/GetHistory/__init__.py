import azure.functions as func
from azure.data.tables import TableServiceClient
from datetime import datetime, timedelta
import configparser
import json

config = configparser.ConfigParser()
config.read('config.ini')
connection_string = config['CONNECTION']['ConnectionString']

table_service = TableServiceClient.from_connection_string(connection_string)
table_client = table_service.get_table_client('parking')


def main(req: func.HttpRequest) -> func.HttpResponse:
    id_str = req.params.get('id')
    if not id_str:
        return func.HttpResponse(
            "Pass the parking lot ID to get its historical data."
        )

    id_name = '_' + id_str
    date_start = (datetime.utcnow()-timedelta(days=1)).strftime('%Y-%m-%d')

    entities = table_client.query_entities(f"PartitionKey ge '{date_start}'", select=['RowKey', id_name])
    entities_list = list(entities)

    times = [x['RowKey'] for x in entities_list]
    data = [int(x[id_name]) for x in entities_list]

    series_pairs = dict(zip(times, data))
    series_str = json.dumps(series_pairs)

    return func.HttpResponse(
        series_str,
        mimetype="application/json"
    )
