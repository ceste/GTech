import pandas as pd
import json, os
from pandas.io.json import json_normalize

from flask import Flask, jsonify, request
import gunicorn

import pickle
import config, utils

from prediction import Prediction

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])
def predict():

	
	# get data
	json_data = request.get_json(force=True)

	print(json_data)

	# text = json_data['text']
	type_ = json_data['type']	

	if type_ == 'news':

		#convert json as dataframe as the model only accept dataframe
		input_df = pd.json_normalize(json_data)
		input_df = input_df[['text']]
		input_df.rename(columns={'text':'News Title'}, inplace=True)	
		input_df.to_excel('news_data/json_data.xls',sheet_name='Data Train')

		prediction = Prediction('news_data/json_data.xls',type_)
		label = prediction.predict()

	elif type_ == 'comment':		

		#convert json as dataframe as the model only accept dataframe
		input_df = pd.json_normalize(json_data)
		input_df = input_df[['text']]
		input_df.rename(columns={'text':'Comment'}, inplace=True)	
		input_df.to_excel('comment_data/json_data.xls',sheet_name='Data Train')		

		prediction = Prediction('comment_data/json_data.xls',type_)
		label = prediction.predict()

	
	result = {'prediction': label}	
	output = result

	
	return jsonify(results=output)

if __name__ == '__main__':
	port = int(os.environ.get("PORT", 5000))
	app.run(port = port, debug=True)

