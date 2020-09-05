import config, utils

import os
import pandas as pd
import numpy as np
import pickle
import re

from pathlib import Path

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, average_precision_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from compress_pickle import dump, load
import io

class Pipeline:
	
	
	def __init__(self, path_to_dataset, model, random_state=42, test_size=0.2):
		
		self.path_to_dataset = path_to_dataset

		self.type_ = model		
		if self.type_ == 'news':
			self.path_to_folder = 'news_data'			
			self.text_field = 'News Title'			
			self.target_column = 'Category'
			self.target_column_mapping = None
		elif self.type_ == 'comment':
			self.path_to_folder = 'comment_data'
			self.text_field = 'Comment'			
			self.target_column = 'Class'
			self.target_column_mapping = {0: "not spam", 1: "spam"}

		if not os.path.exists(self.path_to_folder):
			os.makedirs(self.path_to_folder)

		
		self.random_state = random_state		
		self.test_size = test_size			
		
		xls = pd.ExcelFile(self.path_to_dataset)
		self.dataframe = xls.parse('Data Train')				

		# if self.type_ == 'comment':
		# 	self.dataframe[self.target_column] = np.where(self.dataframe[self.target_column]==0,'Not Spam','Spam')
		
		self.X_train = None
		self.X_valid = None
		self.y_train = None
		self.y_valid = None				

		self.label_encoders = None		
		file = Path(self.path_to_folder,'label_encoders.pickle')
		if file.is_file():
			self.label_encoders = self.load_pickle(file)    		

		self.count_vect = None
		file = Path(self.path_to_folder,'count_vect.pickle')
		if file.is_file():
			self.count_vect = self.load_pickle(file)    		

		self.tfidf_transformer = None		
		file = Path(self.path_to_folder,'tfidf_transformer.pickle')
		if file.is_file():
			self.tfidf_transformer = self.load_pickle(file)    		
		
		self.model = None		
		file = Path(self.path_to_folder,'rf.pickle')		
		if file.is_file():
			self.model = self.load_pickle(file)

		file = Path(self.path_to_folder,'rf_compressed.pickle')		
		if file.is_file():
			self.model = utils.load_compressed_files(file)

		
		

	def split_dataframe(self):

		# if not self.target_column_mapping is None:
		# 	self.dataframe[self.target_column].replace(self.target_column_mapping, inplace=True)

		feature_names = [col for col in self.dataframe.columns if col!=self.target_column]	

		data = self.dataframe.copy()

		X = data[feature_names]
		y = data[self.target_column]
		
		self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

		
		pd.concat([self.X_train, self.y_train], axis=1).to_csv(self.path_to_folder+'/train.csv')
		pd.concat([self.X_valid, self.y_valid], axis=1).to_csv(self.path_to_folder+'/valid.csv')

		
	def load_pickle(self, filename):

		file = open(filename,'rb')
		object_file = pickle.load(file)
		file.close()
		return object_file

	def save_as_pickle(self,filename,obj):
				
		with open(filename, 'wb') as fp:
			pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
		fp.close()


	def encode_target_feature(self,y_train, y_valid):		

		if not self.target_column_mapping is None:
			
			y_train.replace(self.target_column_mapping, inplace=True)
			y_valid.replace(self.target_column_mapping, inplace=True)
	

		self.label_encoders = {}
		self.label_encoders[self.target_column] = LabelEncoder()        
		self.label_encoders[self.target_column].fit(y_train)  
		
		y_train = self.label_encoders[self.target_column].transform(y_train)
		y_valid = self.label_encoders[self.target_column].transform(y_valid)

		
		self.save_as_pickle(self.path_to_folder+'/label_encoders.pickle',self.label_encoders)

		return y_train, y_valid


	def train_model(self, X, y):

		rf = RandomForestClassifier(random_state=self.random_state)
		rf.fit(X, y)
		

		self.model = rf

		#save model		
		self.save_as_pickle(self.path_to_folder+'/rf.pickle',self.model)		
		utils.compress_files(Path(self.path_to_folder,'rf_compressed.pickle'),self.model)



	def predict(self, data):

		return self.model.predict(data)

	def evaluate_model(self, actual, predictions):

		print(confusion_matrix(actual, predictions))
		print("Accuracy:",accuracy_score(actual, predictions))
		print(classification_report(actual, predictions))

	def strip_character(self, dataCol):

		r = re.compile(r'[^a-zA-Z !@#$%&*_+-=|\:";<>,./()[\]{}\']')		
		return r.sub('', dataCol)

	def remove_numbers(self, dataCol):
		
		return re.sub(r'\d+', '', dataCol)

	def remove_non_words(self, dataCol):

		return re.sub(r"\W", " ", dataCol, flags=re.I)

	def stemming_lemmatization(self, data):
	
		wordnet_lemmatizer = WordNetLemmatizer()
		
		nrows = len(data)
		lemmatized_text_list = []

		for row in range(0, nrows):

			# Create an empty list containing lemmatized words
			lemmatized_list = []

			# Save the text and its words into an object
			text = data.iloc[row]['Content_Parsed_4']
			text_words = text.split(" ")

			# Iterate through every word to lemmatize
			for word in text_words:
				lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

			# Join the list
			lemmatized_text = " ".join(lemmatized_list)

			# Append to the list containing the texts
			lemmatized_text_list.append(lemmatized_text)
		
		data['Content_Parsed_5'] = lemmatized_text_list
		
		return data

	def remove_stop_words(self, data):
	
		stop_words = list(stopwords.words('english'))
		
		data['Content_Parsed_6'] = data['Content_Parsed_5']

		for stop_word in stop_words:

			regex_stopword = r"\b" + stop_word + r"\b"
			data['Content_Parsed_6'] = data['Content_Parsed_6'].str.replace(regex_stopword, '')
			
		return data

	def remove_single_character(self, dataCol):

		return re.sub(r"\s+[a-zA-Z]\s+", " ", dataCol)

	def text_cleaning(self, data):

		# Remove possesive pronouns first
		data['Content_Parsed_1'] = data[self.text_field].str.replace("'s", "")
		# add space for $ so it can be considered as feature
		data['Content_Parsed_1'] = data['Content_Parsed_1'].str.replace("$", " money ")

		# Remove special characters
		data['Content_Parsed_2'] = data['Content_Parsed_1'].str.replace("\t", " ")
		data['Content_Parsed_2'] = data['Content_Parsed_2'].str.replace("\n", " ")
		data['Content_Parsed_2'] = data['Content_Parsed_2'].str.replace('"', '')
		data['Content_Parsed_2'] = data['Content_Parsed_2'].apply(self.strip_character)
		data['Content_Parsed_2'] = data['Content_Parsed_2'].apply(self.remove_numbers)
		data['Content_Parsed_2'] = data['Content_Parsed_2'].apply(self.remove_non_words)
		data['Content_Parsed_2'] = data['Content_Parsed_2'].apply(self.remove_single_character)		
		

		# Remove left special characters, overlap with above but never mind, no harm
		data['Content_Parsed_3'] = data['Content_Parsed_2']
		additional_special_characters = list("!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~")

		for item in additional_special_characters:
			data['Content_Parsed_3'] = data['Content_Parsed_3'].str.replace(item, '')

		#Lower the text
		data['Content_Parsed_4'] = data['Content_Parsed_3'].str.lower()

		data = self.stemming_lemmatization(data)

		data = self.remove_stop_words(data)
				
		data.rename(columns={'Content_Parsed_6': 'Content_Parsed'}, inplace=True)		

		return data

	def text_representation(self, data):
		
		if self.count_vect is None:
			self.count_vect = CountVectorizer()
			data_counts = self.count_vect.fit_transform(data['Content_Parsed'])
		else:
			data_counts = self.count_vect.transform(data['Content_Parsed'])

		if self.tfidf_transformer is None:
			self.tfidf_transformer = TfidfTransformer()
			data_tfidf = self.tfidf_transformer.fit_transform(data_counts)
		else:
			data_tfidf = self.tfidf_transformer.transform(data_counts)
		

		self.save_as_pickle(self.path_to_folder+'/count_vect.pickle',self.count_vect)
		self.save_as_pickle(self.path_to_folder+'/tfidf_transformer.pickle',self.tfidf_transformer)

		return data_counts, data_tfidf


	def train(self):		

		self.split_dataframe()		

		self.X_train = self.text_cleaning(self.X_train)
		self.X_valid = self.text_cleaning(self.X_valid)
				
		self.y_train, self.y_valid = self.encode_target_feature(self.y_train, self.y_valid)

		X_train_counts, X_train_tfidf = self.text_representation(self.X_train)
		X_valid_counts, X_valid_tfidf = self.text_representation(self.X_valid)			

		self.train_model(X_train_tfidf, self.y_train)			

		predictions = self.predict(X_valid_tfidf)

		self.evaluate_model(self.y_valid,predictions)

		