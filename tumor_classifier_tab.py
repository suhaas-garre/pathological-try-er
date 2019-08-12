import numpy as np
import pandas as pd
from fastai import *
import preprocessing

#import pre-processed training, testing, and validation datasets
# train, val, test = preprocessing.preprocess("/Users/suhaas/Desktop/stott_lab/joao/HN2_Suhaas_wupdated25.csv")

#assign dependent variable 
dep_var = 'Classifier.Label'

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
