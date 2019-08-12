import numpy as np
import pandas as pd

def preprocess(filePath):
	df = pd.read_csv(filePath)

	# rename terrible column names
	df = df.rename(columns = {'p.EMT.triple.positive':'pEMT_TriplePositive',
							'p.EMT.LAMC2.LAMB3.': 'LAMC2_LAMB3',
							'pEMT.LAMC2.PDPN.': 'LAMC2_PDPN',
							'pEMT.LAMB3.PDPN.': 'LAMB3_PDPN',
							'pEMT.Triple.p63': 'Triple_p63',
							'pEMT.LAMC2.LAMB3.p63.': 'LAMC2_LAMB3_p63',
							'pEMT.LAMC2.PDPN.p63.': 'LAMC2_PDPN_p63',
							'pEMT.LAMB3.PDPN.p63.': 'LAMB3_PDPN_p63',
							'Dapi.Nucleus.Intensity':'Dapi_Intensity', 
							'Opal.520.Cytoplasm.Intensity': 'Opal520', 
							'Opal.540.Cytoplasm.Intensity': 'Opal540',
							'Opal.570.Cytoplasm.Intensity': 'Opal570',
							'Opal.620.Cytoplasm.Intensity': 'Opal620',
							'Opal.650.Nucleus.Intensity': 'Opal650',
							'Opal.690.Cytoplasm.Intensity': 'Opal690',
							'Cell.Area..µm².':'Cell_Area',
							'Cytoplasm.Area..µm².':'Cytoplasm_Area', 
							'Nucleus.Area..µm².':'Nucleus_Area', 
							'Nucleus.Perimeter..µm.':'Nucleus_Perimeter', 
							'Nucleus.Roundness':'Nuclear_Roundness'})

	coor_norm = ['XMin', 'XMax', 'YMin', 'YMax']
	df[coor_norm] = df[coor_norm].astype('float64')

	# columns to drop out of dataframe
	toDrop = ['Unnamed: 0', 'Object.Id', 'PathState', 'Grade', 'pEMT_scRNASeq', 'pEMT_lowhigh']

	# removing unnecessary columns
	df = df.drop(toDrop, axis=1)

	# batch normalize spatial features across all tumors 
	# (cell area, nuclear & cytoplasm roundness/area/perimeter)
	batch_norm = ['Cell_Area', 'Cytoplasm_Area', 'Nucleus_Area', 'Nucleus_Perimeter', 'Nuclear_Roundness']
	df[batch_norm] = df[batch_norm].apply(lambda x: x/np.max(x))

	# split each tumor into its own dataframe
	HN5 = df.loc[df['Patient'] == 'HN5']
	HN6 = df.loc[df['Patient'] == 'HN6']
	HN16 = df.loc[df['Patient'] == 'HN16']
	HN17 = df.loc[df['Patient'] == 'HN17']
	HN18 = df.loc[df['Patient'] == 'HN18']
	HN20 = df.loc[df['Patient'] == 'HN20']
	HN22 = df.loc[df['Patient'] == 'HN22']
	HN25 = df.loc[df['Patient'] == 'HN25']
	HN26 = df.loc[df['Patient'] == 'HN26']
	HN28 = df.loc[df['Patient'] == 'HN28']

	
	df_list = [HN5, HN6, HN16, HN17, HN18, HN20, HN22, HN25, HN26, HN28]

	def normalize(dataframe):
		coor_norm = ['XMin', 'XMax', 'YMin', 'YMax']
		dataframe[coor_norm] = dataframe[coor_norm].apply(lambda y: y - np.min(y))
		dataframe[coor_norm] = dataframe[coor_norm].apply(lambda n: n/np.max(n))

		fluor_norm = ['Dapi_Intensity', 'Opal520', 'Opal540', 'Opal570', 'Opal620', 'Opal650', 'Opal690']
		dataframe[fluor_norm] = dataframe[fluor_norm].apply(lambda y: y/np.max(y))
		return dataframe

	new_list = [normalize(i) for i in df_list]

	#df_normalized = pd.concat(df_list)
	
	train_set = pd.concat([HN6, HN26, HN25, HN5, HN28])
	val_set = pd.concat([HN16, HN17, HN22])
	test_set = pd.concat([HN18, HN20])
	
	return train_set, val_set, test_set


training, validation, testing = preprocess("/Users/suhaas/Desktop/stott_lab/joao/HN2_Suhaas_wupdated25.csv")

