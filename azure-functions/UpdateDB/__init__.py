import logging
import azure.functions as func
from azure.data.tables import TableServiceClient
from azure.data.tables import TableEntity
from datetime import datetime
from urllib.request import urlopen
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
connection_string = config['CONNECTION']['ConnectionString']

table_service = TableServiceClient.from_connection_string(conn_str=connection_string)

url = 'https://data.ntpc.gov.tw/api/datasets/E09B35A5-A738-48CC-B0F5-570B67AD9C78/csv/file'


def update_db():
	response = urlopen(url)
	date_str = datetime.utcnow().strftime('%Y-%m-%d')
	time_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
	df = pd.read_csv(response)

	task = TableEntity()
	task['PartitionKey'] = date_str
	task['RowKey'] = time_str

	for _, data in df.iterrows():
		column = '_' + str(data['ID'])
		value = str(data['AVAILABLECAR'])
		task[column] = value

	table_client = table_service.get_table_client(table_name='parking')
	table_client.create_entity(entity=task)


def main(mytimer: func.TimerRequest) -> None:
    update_db()
