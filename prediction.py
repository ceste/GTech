import config, utils
import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from pipeline import Pipeline



class Prediction(Pipeline):
	
	def __init__(self, path_to_dataset, model):		
		
		Pipeline.__init__(self, path_to_dataset, model)

		self.X = self.dataframe
		


	def decode_label_encoder(self, data):		

		return self.label_encoders[self.target_column].inverse_transform(data)



	def predict(self):
			
		self.X = self.text_cleaning(self.X)		

		X_counts, X_tfidf = super(Prediction, self).text_representation(self.X) 					

		prediction_code = super(Prediction, self).predict(X_tfidf)
		
		prediction_label = self.decode_label_encoder(prediction_code)		
			
		return prediction_label[0]