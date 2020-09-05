import pandas as pd
import numpy as np

from pathlib import Path
import argparse

import json
from pandas.io.json import json_normalize

# import config
# import utils
from pipeline import Pipeline
# from prediction import Prediction
# from log import Log


if __name__ == '__main__':

	model = 'news'
	if model == 'news':
		dataset_path = Path('News Title.xls')

	if dataset_path.exists():		

		print('Training for '+model+' is started!!')		
	
		pipeline = Pipeline(path_to_dataset=dataset_path, test_size=0.2, model=model)
		pipeline.train()

		print('Training for '+model+' is done!!')		

	else:
		print('file for '+model+' is not exist')


	print()

	model = 'comment'
	if model == 'comment':
		dataset_path = Path('Comment Spam.xls')

	if dataset_path.exists():		

		print('Training for '+model+' is started!!')		
	
		pipeline = Pipeline(path_to_dataset=dataset_path, test_size=0.2, model=model)
		pipeline.train()

		print('Training for '+model+' is done!!')		

	else:
		print('file for '+model+' is not exist')

