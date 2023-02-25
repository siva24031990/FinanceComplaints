#To import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numbers
from datetime import datetime
import data_loading
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sys import exit

#from sklearn.preprocessing import SimpleImputer
#from sklearn.model_selection import train_test_split

#To test if sufficient/necessary data is there to proceed
if data_loading.dict2df.consumer_disputed.nunique() == 1:
    #exit("insufficient target data to proceed, aborting")
 
    data_loading.dict2df.loc[0:len(data_loading.dict2df)/2, "consumer_disputed"]="Yes"
    data_loading.dict2df.loc[len(data_loading.dict2df)/2:len(data_loading.dict2df), "consumer_disputed"]="No"

#To Replace empty strings with None in entire df
df_len=len(data_loading.dict2df)
data_loading.dict2df.replace("",None, inplace=True)

#To delete the columns with missing  values more than 30 percent
delete_columns=[]
need_impute=[]
with pd.option_context('display.max_rows',None, 'display.max_columns',None):
    for col in list(data_loading.dict2df.columns):
        print(col,': ', data_loading.dict2df[col][2],end='\n')
        NaN_Percent=data_loading.dict2df[col].isna().sum()/df_len
        print(col, "Nan Percent: ", NaN_Percent)

        if NaN_Percent > 0.3:
            del data_loading.dict2df[col]
            print(col, "is deleted as NaN is more than 30 percent")
            delete_columns.append(col)
        if 0.3 > NaN_Percent > 0:
            print(col,"needs impute")
            need_impute.append(col)

#To delete the rows with missng  values more than 30 percent
new_Coloumn_count = len(list(data_loading.dict2df.columns))
deleted_row=[]
rows_with_NaN=[]
for i in range(df_len):
    row_NaN = data_loading.dict2df.iloc[i].isnull().sum()
    if row_NaN>1:
        rows_with_NaN.append((i,row_NaN))
    if row_NaN/new_Coloumn_count > 0.3 :
        data_loading.dict2df.drop(i)
        deleted_row.append(i)
print("Deleted row count:", len(deleted_row))
print("rows_with_NaN count:", len(rows_with_NaN))
print("rows_with_NaN:", rows_with_NaN)

#To delete the unrequired columns/features based on manual review by checking its unique values
#To delete ID columns as not usefull here: based on maual review
del data_loading.dict2df["complaint_id"]
try:
    print("first complaint ID:", data_loading.dict2df["complaint_id"][0])
except:
    delete_columns.append("complaint_id")
    print("complaint ID deleted")

#To analyse datetime column for useful data and exclude: based on maual review
print("date anlysis")
#print(data_loading.dict2df[data_loading.dict2df['date_sent_to_company']!=data_loading.dict2df['date_received']][['date_sent_to_company','timely','submitted_via']])
print(data_loading.dict2df[data_loading.dict2df['timely']!='Yes'][['date_sent_to_company','date_received','submitted_via']])
#datetime.strptime(date_string, format)
data_loading.dict2df["date_sent_to_company"]=pd.to_datetime(data_loading.dict2df["date_sent_to_company"], format="%Y-%m-%d")
data_loading.dict2df["date_received"]=pd.to_datetime(data_loading.dict2df["date_received"], format="%Y-%m-%d")
data_loading.dict2df["days_from_received_to_sent"]=(data_loading.dict2df["date_sent_to_company"]-data_loading.dict2df["date_received"]).dt.days
print(data_loading.dict2df["days_from_received_to_sent"][0])
del data_loading.dict2df["date_sent_to_company"]
del data_loading.dict2df["date_received"]
delete_columns.append("date_sent_to_company")
delete_columns.append("date_received")
print("deleted date sent company and date received cols, added column days_from_received_to_sent")

#To evaluate zipcode column and reduce dimensionality: based on maual review
zipcode2 = [None]*df_len
for m in range(df_len):
    if data_loading.dict2df['zip_code'][m]:
        zipcode2[m]=data_loading.dict2df['zip_code'][m]    ##[0:3]
    else:
        zipcode2[m]=111
df_zipcode2=pd.DataFrame(zipcode2,columns=['zip_code2'])
print("modified zip:")
temp_var=df_zipcode2['zip_code2'].value_counts()
print(len(temp_var[temp_var.values<35]))
#Ziptoreplace=[]
ziptoreplace=list((temp_var[temp_var.values<35].index))
print(len(ziptoreplace))

for zips in ziptoreplace:
    data_loading.dict2df["zip_code"].replace(zips,"11111",inplace=True)
print(data_loading.dict2df["zip_code"].value_counts())
#print(type(data_loading.dict2df["zip_code"][0]))

#To find and fix outliers for numerical columns

print(len(data_loading.dict2df._get_numeric_data().value_counts()))
for new_col in data_loading.dict2df.columns:
    print(type(data_loading.dict2df[new_col][0]), new_col)
    if isinstance(data_loading.dict2df[new_col][0], numbers.Number):
        q1, q3 = data_loading.dict2df[new_col].quantile([0.25,0.75])  ##for numeric cols
        iqr=q3-q1
        if iqr == 0:
            print(new_col, "has IQR zero, outlier cannot be found")
            break 
        lowest = q1 - 1.5*iqr
        highest = q3 + 1.5*iqr
        print(lowest, highest)
        print(data_loading.dict2df.loc[data_loading.dict2df[new_col]>(q1 - 1.5*iqr), new_col])
        #data_loading.dict2df.loc[data_loading.dict2df[new_col]>(q3 + 1.5*iqr), new_col] = highest
        #data_loading.dict2df.loc[data_loading.dict2df[new_col]<(q1 - 1.5*iqr), new_col] = lowest
print(data_loading.dict2df._get_numeric_data().value_counts())

plt.boxplot(data_loading.dict2df["days_from_received_to_sent"])
plt.show()

#To impute for the null values for required columns/rows
print("Deleted columns", delete_columns)
print("columns needs impute", need_impute)
imputer=SimpleImputer(missing_values=None, strategy="most_frequent")

'''
imputed_zip_code=imputer.fit_transform(data_loading.dict2df['zip_code'].values.reshape(-1,1))
data_loading.dict2df["imputed_zip_code"]=pd.DataFrame(imputed_zip_code)
print(data_loading.dict2df[["zip_code","imputed_zip_code"]])
'''
imputed_columns=[]
for imp in need_impute:
    imputed_columns.append('imputed_'+imp)
print("imputed columns are:", imputed_columns)

for val, impu in zip(need_impute,imputed_columns):
    
    tempu=imputer.fit_transform(data_loading.dict2df[val].values.reshape(-1,1))
    data_loading.dict2df[impu]=pd.DataFrame(tempu)
data_loading.dict2df.drop(columns=need_impute, inplace=True)
print(data_loading.dict2df.columns)


#To scalarise and normalise the data
scalarise=StandardScaler()
ohe=OneHotEncoder()
le=LabelEncoder()
num_features=scalarise.fit_transform(data_loading.dict2df["days_from_received_to_sent"].values.reshape(-1,1))
target_col=le.fit_transform(data_loading.dict2df["consumer_disputed"].values.reshape(-1,1))
print("num_features")
print(np.unique(num_features))
print(type(num_features))

print(data_loading.dict2df["consumer_disputed"].value_counts())
print("target  column")
print(np.unique(target_col))
print(type(target_col))

data_loading.dict2df.drop(columns=["days_from_received_to_sent", "consumer_disputed"], inplace=True)
print(data_loading.dict2df.columns)

ohevector=ohe.fit_transform(data_loading.dict2df.to_numpy())
print(ohevector.get_shape())
print(type(ohevector))
features=np.concatenate((ohevector.toarray(), num_features), 1)
print(features.shape)
print(type(features))

#To reduce dimensionality and balance data

#To Split train and test
X_train, X_test, Y_train, Y_test = train_test_split(features, target_col, test_size=0.25, random_state=42)


'''
#print(data_loading.dict2df.info())
#print(len(data_loading.web_data))  #no of records in df
#print(type(data_loading.web_data["_source"][2])) #top n records of df
#dict2df_columns=list(data_loading.web_data["_source"][0].keys())
#print(dict2df_columns)
#dict2df=pandas.DataFrame.from_records(data_loading.web_data["_source"])
#for n in range(0, 10):
#    dict2df.append(data_loading.web_data["_source"][n],ignore_index=True)
#print(data_loading.dict2df.head(10))
#print(data_loading.dict2df.columns)

#print(data_loading.web_data.columns.values)
#column_names=data_loading.web_data.columns.values
#for col in data_loading.web_data.columns.values:
    #if col == "_score":
    #print(data_loading.web_data.groupby(col)[col].count()[0])  #columnwise Unique values count
    #print(data_loading.web_data[col].groupby(data_loading.web_data[col].values)[data_loading.web_data[col].values].count())

#print(data_loading.web_data["_score"].isnull().sum())  #coloumn null values count
#print(data_loading.web_data.isnull().sum().values[1])  #entire df columnwise null counts values
#print(data_loading.web_data.isnull().sum().index[1])   # access the entire df null counts index

#if data_loading.dict2df['zip_code'].value_counts()>1 is True:
#print("new_Coloumn_count:", new_Coloumn_count)
#print(len(data_loading.dict2df['zip_code']))

#print("old df len :",df_len, "new df len:", len(data_loading.dict2df))
'''