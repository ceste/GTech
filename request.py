import requests
import json
import config, utils
import argparse
import pandas as pd
from pathlib import Path
from prediction import Prediction

def pass_to_api(url, data):

	r_survey = requests.post(url, data)
	send_request = requests.post(url, data)	

	if send_request.status_code == 200:

		print(send_request.json()['results'])
	else:
		print('There is an error occurs')


	


if __name__ == '__main__':

	# # local url
	url = config.LOCAL_URL

	# news

	# data dummy	
	data_dummy = '{"text" :"Google shows off Androids for wearables, cars, and TVs", "type":"news"}'

	json_data = json.loads(data_dummy)	
	
	type_ = json_data['type']

	# print('Predict:',json_data)		

	data = json.dumps(json_data)	


	#try on local 

	# input_df = pd.json_normalize(json_data)	
	# input_df = input_df[['text']]
	# input_df.rename(columns={'text':'News Title'}, inplace=True)	
	# input_df.to_excel('news_data/json_data.xls',sheet_name='Data Train')	

	# prediction = Prediction('news_data/json_data.xls',type_)
	# label = prediction.predict()

	# print(label)

	# end of try on local 
	
	pass_to_api(url, data)

	print()

	# comment

	# data dummy	
	data_dummy = '{"text" :"Find out how i make $20 Million/year online with these easy steps !", "type":"comment"}'

	json_data = json.loads(data_dummy)	
	
	type_ = json_data['type']

	# print('Predict:',json_data)		
	# print('type_:',type_)

	data = json.dumps(json_data)	

	pass_to_api(url, data)

	#try on local 

	# input_df = pd.json_normalize(json_data)	
	# input_df = input_df[['text']]
	# input_df.rename(columns={'text':'Comment'}, inplace=True)	
	# input_df.to_excel('comment_data/json_data.xls',sheet_name='Data Train')		

	# prediction = Prediction('comment_data/json_data.xls',type_)
	# label = prediction.predict()

	# print(label)

	# end of try on local 

	