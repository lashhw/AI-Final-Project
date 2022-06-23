import logging
import azure.functions as func
from azure.data.tables import TableServiceClient
from azure.data.tables import TableEntity
from datetime import datetime, timedelta
from urllib.request import urlopen
import pandas as pd
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
connection_string = config['CONNECTION']['ConnectionString']

ids_train = np.load('./models/common/files/ids_train.npy')

table_service = TableServiceClient.from_connection_string(conn_str=connection_string)
table_client = table_service.get_table_client(table_name='parking')

url = 'https://data.ntpc.gov.tw/api/datasets/E09B35A5-A738-48CC-B0F5-570B67AD9C78/csv/file'


def main(mytimer: func.TimerRequest) -> None:
	response = urlopen(url)
	date_str = datetime.utcnow().strftime('%Y-%m-%d')
	time_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
	df = pd.read_csv(response)

	task = TableEntity()
	task['PartitionKey'] = date_str
	task['RowKey'] = time_str

	for _, data in df.iterrows():
		if data['ID'] in ids_train:
			column = '_' + str(data['ID'])
			value = str(data['AVAILABLECAR'])
			task[column] = value

	table_client.create_entity(entity=task)

	old_date_str = (datetime.utcnow()-timedelta(days=7)).strftime('%Y-%m-%d')
	old_time_str = (datetime.utcnow()-timedelta(days=7)).strftime('%Y-%m-%d %H:%M')
	old_query_string = f"PartitionKey le '{old_date_str}' and RowKey lt '{old_time_str}'"
	old_entities = table_client.query_entities(old_query_string, 
											   select=['PartitionKey', 'RowKey'])
	for e in old_entities:
		table_client.delete_entity(e['PartitionKey'], e['RowKey'])