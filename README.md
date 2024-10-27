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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('Placement_Data_Full_Class.csv')
dataset.head()
```

### Output:
![Screenshot 2024-10-09 103951](https://github.com/user-attachments/assets/e3aad698-af06-43a4-bdc5-8a34bb88ca6a)


```
dataset.info()
```
### Output:
![Screenshot 2024-10-09 104012](https://github.com/user-attachments/assets/8820330f-7e1e-48da-86d6-d317a106fbbb)


```
dataset = dataset.drop('sl_no', axis=1);
dataset.info()
```
### Output:
![Screenshot 2024-10-09 104018](https://github.com/user-attachments/assets/2ce27a91-ad3d-4ab7-9fee-317e15627700)


```
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
```
### Output:
![Screenshot 2024-10-09 104023](https://github.com/user-attachments/assets/25000505-014a-4957-a3f0-9dcdf86b0cfa)


```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

```
### Output:
![Screenshot 2024-10-09 104032](https://github.com/user-attachments/assets/5ec2f02d-e760-40c4-9f1c-f2eaf3de176e)

```
x = dataset.iloc[:,:-1]
x
```
### Output:
![Screenshot 2024-10-09 104041](https://github.com/user-attachments/assets/a002ddd8-8164-4253-a6f5-00c8adc5b0f7)

```
y=dataset.iloc[:,-1]
y
```
### Output:
![Screenshot 2024-10-09 104049](https://github.com/user-attachments/assets/055eb2cb-d880-4e0d-85f9-ee501596ab6c)

```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
y_pred
```
### Output:
![Screenshot 2024-10-09 104055](https://github.com/user-attachments/assets/b42a31f1-a7a1-413e-92ad-e07e70259243)

```
from sklearn.metrics import accuracy_score, confusion_matrix
cf = confusion_matrix(y_test, y_pred)
cf
```
### Output:
![Screenshot 2024-10-09 104101](https://github.com/user-attachments/assets/1319dc33-2d63-4df5-a740-cb6925222a0b)

```
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
### Output:
![Screenshot 2024-10-09 104106](https://github.com/user-attachments/assets/bed5a224-20f6-491c-a30b-812fabd02750)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
