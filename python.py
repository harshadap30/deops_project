import numpy as np 
import pandas as pd
company = pd.read_csv('/data/1000_Companies.csv')
x =company.iloc[:,:-1].values
y = company.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
x[:,3] = encoder.fit_transform(x[:,3])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
y_pred = pd.DataFrame(y_pred)
y_test = pd.DataFrame(y_test, columns=['original'])
y_test['prediction'] = y_pred

y_test.to_csv('/output/Result.csv',index=False)
