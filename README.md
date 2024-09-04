# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mohamed Anas O.I
RegisterNumber: 212223110028
*/
```

```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df.info()
```
![Screenshot 2024-09-04 135408](https://github.com/user-attachments/assets/2863c8f1-660c-4c4b-88f8-e0716d91bd6e)

```
X = df.drop(columns=["AveRooms", "AveBedrms"])
X.info()
```
![Screenshot 2024-09-04 135415](https://github.com/user-attachments/assets/dae81bdd-28f3-4aa8-87ec-c02170b03df1)

```
Y = df[["AveRooms", "AveBedrms"]]
Y.info()
```
![Screenshot 2024-09-04 135425](https://github.com/user-attachments/assets/978e07e7-004a-4082-a4f2-51bd911312a6)

```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

print(X_train)
```
![Screenshot 2024-09-04 135433](https://github.com/user-attachments/assets/15c06352-0837-40b7-b961-5e469fa31fb7)

```
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

multi_output_sgd = MultiOutputRegressor(sgd)

multi_output_sgd.fit(X_train, Y_train)

Y_pred = multi_output_sgd.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

print("\nPredictions:\n", Y_pred[:5])
```

![Screenshot 2024-09-04 135439](https://github.com/user-attachments/assets/3ebc0f7c-469e-4d6c-bad7-715e2275ac59)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
