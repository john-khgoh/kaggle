from os import getcwd
from statistics import mode
import pandas as pd
import numpy as np
from math import isnan
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

#pd.set_option('display.max_rows', 4000)
#pd.set_option('display.max_columns', 100)

#Scoring function
def NumPyRMSLE(y_true:list, y_pred:list) -> float:
    n = len(y_true)
    msle = np.mean([(np.log(y_pred[i] + 1) - np.log(y_true[i] + 1)) ** 2.0 for i in range(n)])
    return np.sqrt(msle)

#Custom encoder that encodes alphabetically 
#It takes the main dataframe to get the full range of categorical data, and encodes observing legal NAs
def cust_encoder(main_df):
	main_df_keys = main_df.keys()
	encoded_col_list = []
	full_encoder_list = []
	for _ in main_df_keys: #Iterate through columns
		datatype = main_df[_].dtype #Get the datatype for a given column
		#print("%s:%s" %(_,datatype))
		data_list = []
		encoder_dict_list = []
		if(datatype=="object"): #Only encode categorical (string) data
			data_list = list(main_df[_])
			cg = list(set(data_list)) #Get categories
			try:
				#legal_na_list.index(main_df[_]) #Checks for membership of legal_na_list, otherwise leaves np.nan as is 
				cg.remove(np.nan) #Replacing np.nan with string nan for alphabetical sorting
				cg.append("nan")
			except:
				pass
			cg.sort()
			for i in range(len(cg)): #Iterate through unique elements in a column
				#encode and record
				if((cg[i]=="nan") & (legal_na_list.count(_)==0)): #Checks for nan and if it's NOT a member
					main_df[_] = main_df[_].replace(np.nan,"nan")
					continue
				elif((cg[i]=="nan") & (legal_na_list.count(_)>0)): #Checks for nan and if it IS a member
					main_df[_] = main_df[_].replace(np.nan,int(i+1))
					encoder_dict_list.append({"nan":i+1})
					continue
				else:	
					main_df[_] = main_df[_].replace(cg[i],i+1)
					encoder_dict_list.append({cg[i]:i+1})
					#cg[i] = i
			full_encoder_list.append(encoder_dict_list)
			encoded_col_list.append(_) #Record the name of encoded columns
		else:
			main_df[_] = main_df[_].replace(np.nan,"nan") #Numerical columns
			
	#print(full_encoder_list)
	return main_df

#Converts NAs to statistical mode
def na_to_mode(df): 
	df_keys = df.keys()
	for _ in df_keys:
		element_list = []
		element_list = df[_]
		element_list = [x for x in element_list if x != "nan"]
		try:
			mode_element = mode(element_list)
			df[_] = df[_].replace("nan",mode_element)
		except:
			df[_] = df[_].replace("nan",0)
	return df

#Checking datatype of columns in a dataframe
def df_datatype(df): 
	df_keys = df.keys()
	dtype_list = []
	for _ in df_keys:
		dtype_list.append(df[_].dtype)
	var_df = pd.DataFrame(df_keys,columns=["variables"])
	dtype_df = pd.DataFrame(dtype_list,columns=["dtype"])
	dtype_df = pd.concat([var_df,dtype_df],axis=1)
	return dtype_df

#List of legal NAs, according to the documentation
legal_na_list = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]
	
wd = getcwd()
train_file = wd + "\\train.csv"
test_file = wd + "\\test.csv"
output_file = wd + "\\output.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

train_x_df = train_df.drop(columns=["SalePrice"])

#Combining train and test dataframes for more accurate regression of NAs
comb_df = pd.concat([train_x_df,test_df],axis=0)
comb_df = cust_encoder(comb_df)
comb_df_keys = comb_df.keys()
len_comb_df = len(comb_df)

#Isolating a single column at a time and using the rest for regression to predict and fill NAs
reg = XGBRegressor()
clone_df = comb_df.copy()

for _ in comb_df_keys:	
	predr_df = comb_df[comb_df[_]=="nan"]
	trainr_df = comb_df[comb_df[_]!="nan"]
	if(len(predr_df)==0): #If there are no predictors, skip
		continue
	
	predr_df = na_to_mode(predr_df)
	trainr_df = na_to_mode(trainr_df)
	
	predr_df = predr_df.astype("float64")
	trainr_df = trainr_df.astype("float64")
	
	y_pred = predr_df[_]
	X_pred = predr_df.drop(columns=[_])
	y_train = trainr_df[_]
	X_train = trainr_df.drop(columns=[_])
	try:
		model = reg.fit(X_train, y_train)
	except:
		print("Failed at %s..." %_)
		#X_train.to_csv(wd + "\\X_train.csv")
		#y_train.to_csv((wd + "\\y_train.csv"))
		raise Exception
	pred_y = model.predict(X_pred)
	
	#Mapping the predicted values back using the Id from X_pred
	x_pred_id_list = list(X_pred["Id"])
	x_pred_id_df = pd.DataFrame(x_pred_id_list,columns=["Id"])
	pred_y_df = pd.DataFrame(pred_y,columns=[_])
	pred_y_df = pd.concat([x_pred_id_df,pred_y_df],axis=1)
	
	for i in range(len(pred_y_df)):
		id = pred_y_df.at[i,"Id"]
		value = pred_y_df.at[i,_]
		if(value<0):
			value = 0
		clone_df.loc[clone_df["Id"]==id,[_]] = value

#Mapping clone_df entries back to train_x_df and test_df
train_id_list = list(train_x_df["Id"])
test_id_list = list(test_df["Id"])
clone_id_list = list(clone_df["Id"])
len_clone_df = len(clone_df)

for _ in range(len_clone_df):
	id = clone_id_list[_]
	if(train_id_list.count(id)>0):
		train_x_df.loc[train_x_df["Id"]==id] = clone_df.loc[clone_df["Id"]==id]
	elif(test_id_list.count(clone_id_list[_])>0):
		test_df.loc[test_df["Id"]==id] = clone_df.loc[clone_df["Id"]==id]
	else:
		raise Exception

train_x_df = train_x_df.drop(columns=["Id"])
test_df = test_df.drop(columns=["Id"])
train_x_arr = np.array(train_x_df)

#Train test split and the model
X_train, X_test, y_train, y_test = train_test_split(train_x_arr,train_df['SalePrice'],test_size=0.01)

regs = [
	("gbr", GradientBoostingRegressor()),
    ("xgb", XGBRegressor())
]

pipe = Pipeline([
    ("voting", VotingRegressor(regs))
])

#Check the accuracy of the prediction against 1% of the training data
pipe.fit(X_train, y_train)
predictions_train = pipe.predict(X_test)
#print(NumPyRMSLE(list(y_test),list(predictions_train)))

#Predicts based on the test data and outputs a CSV file
prediction_test = pipe.predict(test_df)
saleprice_df = pd.DataFrame(prediction_test,columns=["SalePrice"])
id_df = pd.DataFrame(test_id_list,columns=["Id"])
saleprice_df = pd.concat([id_df,saleprice_df],axis=1)
#saleprice_df.to_csv(output_file)