import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastai import *
from fastai.tabular import *
from fastai.basic_data import DataBunch

#import pre-processed training, testing, and validation datasets
train = pd.read_csv('/Users/suhaas/Desktop/stott_lab/train.csv')
test = pd.read_csv('/Users/suhaas/Desktop/stott_lab/test.csv')

categorical = ['pEMT_TriplePositive',
			'LAMC2_LAMB3',
			'LAMC2_PDPN',
			'LAMB3_PDPN', 
			'Triple_p63',
			'LAMC2_LAMB3_p63',
			'LAMC2_PDPN_p63',
			'LAMB3_PDPN_p63',
			'Classifier.Label']

train[categorical] = train[categorical].astype('category')
test[categorical] = test[categorical].astype('category')

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


# Training Data Bunch
data = TabularDataBunch.from_df(df=train, 
						path='.', 
						cat_names=cat_var,
						cont_names=cont_var, 
						dep_var="Classifier.Label", 
						test_df=test, 
						valid_idx=list(range(750000, 1272701)))	


learn = tabular_learner(data, layers=[100,50], emb_drop=0.1, metrics=error_rate)

# learn.lr_find()
# learn.recorder.plot()
# plt.show()

learn.fit_one_cycle(1, max_lr=3e-2)

# predictions, *_ = learn.get_preds(DatasetType.Test)
# labels = np.argmax(predictions, 1)