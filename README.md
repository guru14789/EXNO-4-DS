# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
## STEP 1:
Read the given Data.
## STEP 2:
Clean the Data Set using Data Cleaning Process.
## STEP 3:
Apply Feature Scaling for the feature in the data set.
## STEP 4:
Apply Feature Selection for the feature in the data set.
## STEP 5:
Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains the same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is, it divides every observation by the maximum value of the variable. The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1. Filter Method
2. Wrapper Method
3. Embedded Method

# CODING AND OUTPUT:
```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/460ecd1e-d7fa-4800-a419-8b2d2fe98e85)

```py
df.head()
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/23d7be98-2a54-4ed2-9e79-26d4a17f7fa8)

```py
df.dropna()
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/00294f6b-e746-42bb-8b78-7f279f03c262)
```py
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/3c7366f4-a14d-4219-ba22-a2fad3e2b960)
```py
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/85940f5b-7dd9-4798-91a7-8cd94bf5a30c)
```py
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/b02958f5-bca4-4466-a9d4-802b79d170ec)
```py
from sklearn.preprocessing import Normalizer
scale=Normalizer()
df[['Height','Weight']]=scale.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/10890b3f-f9e1-4aea-b5ec-727aebcb1692)
```py
from sklearn.preprocessing import MaxAbsScaler
scalen=MaxAbsScaler()
df[['Height','Weight']]=scalen.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/5e5fcfca-cd57-4b53-abf4-4e126ca25e88)
```py
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/26d66452-a135-4243-8f2d-b47dbb5b8348)
```py
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/2dc0f653-bd2d-45f4-aa2f-0fe6874dfd54)
```py
data.isnull().sum()
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/db907d23-39d7-4abc-b669-9c970ddcd4e6)
```py
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/3401db30-8b6e-4235-aaa1-4f9bc5cd1735)
```py
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/bbf1c633-a912-4ed7-8d9e-4cc7d3d856cb)
```py
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/39f68c82-b6bf-45e4-b126-1110fa6d5e6f)
```py
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/bd3e993b-ccfc-42ea-8ea5-cce7dea1b4fa)
```py
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/0ca392cf-3fbc-4d0f-9292-9704894bbd10)
```py
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/9e2a7349-5ca6-48f7-b398-940e2a38631f)
```py
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/a0c1da5c-18d4-4510-842e-09bc061dbbe0)
```py
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/7b95e4fd-ef1e-407a-ba3f-186bd938a545)
```py
x=new_data[features].values
x
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/e72cd3ff-3f9a-432d-a40e-7596f464787b)
```py
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/450c4d39-4e99-4437-be1f-b34ac64b1525)
```py
prediction=KNN_classifier.predict(test_x)
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/14fcf028-4d87-460e-8757-18728e49ccee)
```py
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/1b6440ef-c6c6-440f-9b86-317586560009)
```py
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/f35864d0-6985-43b3-b821-1538039615a2)
```py
data.shape
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/a6544ea7-8884-4b26-a002-f1ada3f495e1)
```py
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/3d318909-7fbb-4d1b-b696-b5b61abfa88a)
```py
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/82f083f9-49c2-447d-a214-69dfa9ee6375)
```py
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/81020674-0b46-43f9-b4e7-aac8f7a95b46)
```py
chi2,p, _, _ =chi2_contingency(contigency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![image](https://github.com/guru14789/EXNO-4-DS/assets/151705853/6409cb43-d8f0-4340-9a7f-3785a3cfe53b)
# RESULT:
Thus perform Feature Scaling and Feature Selection process and save the data to a file successfully.
