# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1: Start

STEP 2: Import necessary libraries: 'numpy', 'pandas', and 'StandardScaler'.

STEP 3: Define a 'linear_regression' function that initializes 'theta' and performs gradient descent to update theta based on the error between predictions and actual values.

STEP 4: Load the dataset from the CSV file into a Pandas DataFrame

STEP 5: Extract the feature matrix 'x' and target values 'y' from the dataset.

STEP 6: Standardize 'x' and 'y' using StandardScaler to normalize the data.

STEP 7: Call the 'linear_regression' function with standardized data to learn the model parameters 'theta'.

STEP 8: Prepare a new data point, standardize it, and predict the target value using the learned 'theta'.

STEP 9: Output the predicted value, reversing the standardization to return it to the original scale.

STEP 10: End

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: DINESH KUMARAA K 
RegisterNumber: 212222220012 
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        
        #calculate erros
        errors=(predictions-y).reshape(-1,1)
        
        #update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data = pd.read_csv("C:/Users/SEC/Downloads/50_Startups.csv",header=None)
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#Learn model Parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)
#Predict target value for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
Data values

![image](https://github.com/user-attachments/assets/ffeae361-1f7e-4ff1-bd28-6b75a518d2c6)

Predicted values

![image](https://github.com/user-attachments/assets/84b63317-5142-4f36-8bfb-012f431e3282)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
