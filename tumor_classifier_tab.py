import numpy as np
import pandas as pd
from fastai import *
from fastai.tabular import *
from fastai.basic_data import DataBunch

#import pre-processed training, testing, and validation datasets
train = pd.read_csv('/Users/suhaas/Desktop/stott_lab/train.csv')
test = pd.read_csv('/Users/suhaas/Desktop/stott_lab/test.csv')

train['Classifier.Label'] = train['Classifier.Label'].astype('category')
test['Classifier.Label'] = test['Classifier.Label'].astype('category')

print(train)

#assign categorical variables
cat_var = ['pEMT_TriplePositive',
			'LAMC2_LAMB3',
			'LAMC2_PDPN',
			'LAMB3_PDPN', 
			'Triple_p63',
			'LAMC2_LAMB3_p63',
			'LAMC2_PDPN_p63',
			'LAMB3_PDPN_p63']

#assign continuous variables
cont_var = ['Opal520',
			'Opal540',
			'Opal570',
			'Opal620',
			'Opal650',
			'Opal690',
			'Cell_Area',
			'Cytoplasm_Area',
			'Nucleus_Area',
			'Nucleus_Perimeter',
			'Nuclear_Roundness',
			'XMin',
			'XMax',
			'YMin',
			'YMax']

'''
# Training Data Bunch
data = TabularDataBunch.from_df(df=train, 
						path='.', 
						cat_names=cat_var,
						cont_names=cont_var, 
						dep_var="Classifier.Label", 
						test_df=test, 
						valid_idx=list(range(1000000, 1272701)))	

data.show_batch(rows=10)
'''
