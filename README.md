# Exp-02: Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn. 4. Assign the points for representing in the graph.
4. Predict the regression for marks by using the representation of the graph.
5. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
#Program to implement the simple linear regression model for predicting the marks scored.
#Developed by: JANANI.V.S
#RegisterNumber:212222230050
```
~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import libraries to find mae, mse

from sklearn.metrics import mean_absolute_error,mean_squared_error

# Read csv file

df=pd.read_csv("student_scores.csv")

# Displaying the content in datafile

print("HEAD:")
print(df.head())
print("TAIL:")
print(df.tail())

# Segregating data to variables

x=df.iloc[:,:-1].values
print("x values")
print(x)

y=df.iloc[:,1].values
print("y values")
print(y)

# Splitting train and test data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

# Displaying predicted values

print("Predicted values")
print(Y_pred)

# Displaying actual values

print("Actual values")
print(Y_test)

# Graph plot for training data

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours Vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Graph plot for test data

plt.scatter(X_test,Y_test,color="green")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Find MAE,MSE,RMSE

MSE = mean_squared_error(Y_test,Y_pred)
print('MSE = ',MSE)
MAE = mean_absolute_error(Y_test,Y_pred)
print('MAE = ',MAE)
RMSE=np.sqrt(MSE)
print("RMSE = ",RMSE)
~~~

## Output:
![image](https://github.com/user-attachments/assets/9faadd4d-e3f3-4531-bfa5-b7f6af6b4db7)

### Array value of X:
![image](https://github.com/user-attachments/assets/8f3fcb0b-8e79-4b98-badd-bbeb39aadf01)

### Array value of Y:
![image](https://github.com/user-attachments/assets/87bcbec9-0eb0-4f4d-8714-582d3d10185e)

### Values of Y Prediction:
![image](https://github.com/user-attachments/assets/6e9f4b4d-d36b-46aa-be3a-f96d2a62d26d)

### Array values of Y Set:
![image](https://github.com/user-attachments/assets/5f38f6f0-cb29-4db3-8baf-0cb591d835cb)

### Training Set Graph:
![image](https://github.com/user-attachments/assets/20bca265-f574-4e40-8dde-e9519590228b)

### Test Set Graph:
![image](https://github.com/user-attachments/assets/2366250c-da60-401b-a508-632cbbbfa0c0)

### Values of MSE, MAE and RMSE:
![image](https://github.com/user-attachments/assets/b837b47f-df68-4ec9-b126-70824af538c8)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
