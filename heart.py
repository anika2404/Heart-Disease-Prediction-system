import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the csv data into pandas dataframe
heart_data= pd.read_csv('dataset.csv')

#printing first 5 rows of dataset
print(heart_data.head())
#printing last 5 rows of dataset
print(heart_data.tail())

#number of rows and columns in the dataset
print(heart_data.shape)

#getting some info
print(heart_data.info())

#cheching missing values
heart_data.isnull().sum()

#statistical measures about the data
print(heart_data.describe())

#checking the distribution of target variable
print(heart_data['target'].value_counts())

# 1 represents defective heart and 0 represents healthy heart.

#last 5 rows of dataset
print(heart_data.tail())
#splitting features and target
X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']
print(X)
print(Y)

#splitting data
#test size is how much of the data is taken as test data
#stratify evenly distributes the data classes
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

#logistic regression
model=LogisticRegression()
#training the model
model.fit(X_train,Y_train)

#accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy on Training data:",training_data_accuracy)

#accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy on Test data:",test_data_accuracy)

#predictive system 

input_data=(52,1,0,125,212,0,1,168,0,1,2,2,3)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshape)
print(prediction)
    
if (prediction[0]==0):   #[0] means the first value present in the list
        print("Person does not have heart disease")
else:
        print("Person has heart disease")

                    