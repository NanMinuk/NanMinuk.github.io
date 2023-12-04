---
title: Logistic Regression
tags: Machine_Learnig
typora-root-url: ../
---



# 데이터 전처리

사용데이터셋: [Employees Evaluation for Promotion (kaggle.com)](https://www.kaggle.com/datasets/muhammadimran112233/employees-evaluation-for-promotion)




```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
hr_df=pd.read_csv('hr.csv')
```


```python
hr_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>employee_id</th>
      <th>department</th>
      <th>region</th>
      <th>education</th>
      <th>gender</th>
      <th>recruitment_channel</th>
      <th>no_of_trainings</th>
      <th>age</th>
      <th>previous_year_rating</th>
      <th>length_of_service</th>
      <th>awards_won?</th>
      <th>avg_training_score</th>
      <th>is_promoted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65438</td>
      <td>Sales &amp; Marketing</td>
      <td>region_7</td>
      <td>Master's &amp; above</td>
      <td>f</td>
      <td>sourcing</td>
      <td>1</td>
      <td>35</td>
      <td>5.0</td>
      <td>8</td>
      <td>0</td>
      <td>49</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65141</td>
      <td>Operations</td>
      <td>region_22</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>1</td>
      <td>30</td>
      <td>5.0</td>
      <td>4</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7513</td>
      <td>Sales &amp; Marketing</td>
      <td>region_19</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>sourcing</td>
      <td>1</td>
      <td>34</td>
      <td>3.0</td>
      <td>7</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2542</td>
      <td>Sales &amp; Marketing</td>
      <td>region_23</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>2</td>
      <td>39</td>
      <td>1.0</td>
      <td>10</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48945</td>
      <td>Technology</td>
      <td>region_26</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>1</td>
      <td>45</td>
      <td>3.0</td>
      <td>2</td>
      <td>0</td>
      <td>73</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
hr_df.drop('employee_id',axis=1,inplace=True)
```


```python
hr_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 54808 entries, 0 to 54807
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   department            54808 non-null  object 
     1   region                54808 non-null  object 
     2   education             52399 non-null  object 
     3   gender                54808 non-null  object 
     4   recruitment_channel   54808 non-null  object 
     5   no_of_trainings       54808 non-null  int64  
     6   age                   54808 non-null  int64  
     7   previous_year_rating  50684 non-null  float64
     8   length_of_service     54808 non-null  int64  
     9   awards_won?           54808 non-null  int64  
     10  avg_training_score    54808 non-null  int64  
     11  is_promoted           54808 non-null  int64  
    dtypes: float64(1), int64(6), object(5)
    memory usage: 5.0+ MB



```python
hr_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>no_of_trainings</th>
      <th>age</th>
      <th>previous_year_rating</th>
      <th>length_of_service</th>
      <th>awards_won?</th>
      <th>avg_training_score</th>
      <th>is_promoted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>54808.000000</td>
      <td>54808.000000</td>
      <td>50684.000000</td>
      <td>54808.000000</td>
      <td>54808.000000</td>
      <td>54808.000000</td>
      <td>54808.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.253011</td>
      <td>34.803915</td>
      <td>3.329256</td>
      <td>5.865512</td>
      <td>0.023172</td>
      <td>63.386750</td>
      <td>0.085170</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.609264</td>
      <td>7.660169</td>
      <td>1.259993</td>
      <td>4.265094</td>
      <td>0.150450</td>
      <td>13.371559</td>
      <td>0.279137</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>39.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>29.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>51.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>33.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>60.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>39.000000</td>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>76.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.000000</td>
      <td>60.000000</td>
      <td>5.000000</td>
      <td>37.000000</td>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
hr_df.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>department</th>
      <th>region</th>
      <th>education</th>
      <th>gender</th>
      <th>recruitment_channel</th>
      <th>no_of_trainings</th>
      <th>age</th>
      <th>previous_year_rating</th>
      <th>length_of_service</th>
      <th>awards_won?</th>
      <th>avg_training_score</th>
      <th>is_promoted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>54808</td>
      <td>54808</td>
      <td>52399</td>
      <td>54808</td>
      <td>54808</td>
      <td>54808.000000</td>
      <td>54808.000000</td>
      <td>50684.000000</td>
      <td>54808.000000</td>
      <td>54808.000000</td>
      <td>54808.000000</td>
      <td>54808.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>9</td>
      <td>34</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Sales &amp; Marketing</td>
      <td>region_2</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>16840</td>
      <td>12343</td>
      <td>36669</td>
      <td>38496</td>
      <td>30446</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.253011</td>
      <td>34.803915</td>
      <td>3.329256</td>
      <td>5.865512</td>
      <td>0.023172</td>
      <td>63.386750</td>
      <td>0.085170</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.609264</td>
      <td>7.660169</td>
      <td>1.259993</td>
      <td>4.265094</td>
      <td>0.150450</td>
      <td>13.371559</td>
      <td>0.279137</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>39.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>29.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>51.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>33.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>60.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>39.000000</td>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>76.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.000000</td>
      <td>60.000000</td>
      <td>5.000000</td>
      <td>37.000000</td>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
hr_df['education'].value_counts()
```




    Bachelor's          36669
    Master's & above    14925
    Below Secondary       805
    Name: education, dtype: int64



# 데이터 시각화






```python
sns.barplot(x='previous_year_rating',y='is_promoted',data=hr_df)
```




    <AxesSubplot:xlabel='previous_year_rating', ylabel='is_promoted'>




![output_9_1](/images/2023-12-03-LogisticRegression/output_9_1.png)



```python
sns.barplot(x=hr_df['previous_year_rating'],y=hr_df['is_promoted'],palette='Set2')
```




    <AxesSubplot:xlabel='previous_year_rating', ylabel='is_promoted'>




![output_10_1](/images/2023-12-03-LogisticRegression/output_10_1.png)



```python
sns.barplot(x=hr_df['previous_year_rating'],y=hr_df['is_promoted'],palette='Set3')
```




    <AxesSubplot:xlabel='previous_year_rating', ylabel='is_promoted'>




![output_11_1](/images/2023-12-03-LogisticRegression/output_11_1.png)



```python
sns.lineplot(x=hr_df['avg_training_score'],y=hr_df['is_promoted'],palette='Set2')
```




    <AxesSubplot:xlabel='avg_training_score', ylabel='is_promoted'>




![output_12_1](/images/2023-12-03-LogisticRegression/output_12_1.png)



```python
sns.barplot(x=hr_df['recruitment_channel'],y=hr_df['is_promoted'],palette='Set3')
```




    <AxesSubplot:xlabel='recruitment_channel', ylabel='is_promoted'>




![output_13_1](/images/2023-12-03-LogisticRegression/output_13_1.png)



```python
hr_df['recruitment_channel'].value_counts()
```




    other       30446
    sourcing    23220
    referred     1142
    Name: recruitment_channel, dtype: int64




```python
sns.barplot(x=hr_df['gender'],y=hr_df['is_promoted'],palette='Set3')
```




    <AxesSubplot:xlabel='gender', ylabel='is_promoted'>



![output_15_1](/images/2023-12-03-LogisticRegression/output_15_1.png)

```python
sns.barplot(x=hr_df['department'],y=hr_df['is_promoted'],palette='Set3')
plt.xticks(rotation=45)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
     [Text(0, 0, 'Sales & Marketing'),
      Text(1, 0, 'Operations'),
      Text(2, 0, 'Technology'),
      Text(3, 0, 'Analytics'),
      Text(4, 0, 'R&D'),
      Text(5, 0, 'Procurement'),
      Text(6, 0, 'Finance'),
      Text(7, 0, 'HR'),
      Text(8, 0, 'Legal')])




![output_16_1](/images/2023-12-03-LogisticRegression/output_16_1.png)



```python
plt.figure(figsize=(15,10))
sns.barplot(x=hr_df['region'],y=hr_df['is_promoted'],palette='Set3')
plt.xticks(rotation=45)
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]),
     [Text(0, 0, 'region_7'),
      Text(1, 0, 'region_22'),
      Text(2, 0, 'region_19'),
      Text(3, 0, 'region_23'),
      Text(4, 0, 'region_26'),
      Text(5, 0, 'region_2'),
      Text(6, 0, 'region_20'),
      Text(7, 0, 'region_34'),
      Text(8, 0, 'region_1'),
      Text(9, 0, 'region_4'),
      Text(10, 0, 'region_29'),
      Text(11, 0, 'region_31'),
      Text(12, 0, 'region_15'),
      Text(13, 0, 'region_14'),
      Text(14, 0, 'region_11'),
      Text(15, 0, 'region_5'),
      Text(16, 0, 'region_28'),
      Text(17, 0, 'region_17'),
      Text(18, 0, 'region_13'),
      Text(19, 0, 'region_16'),
      Text(20, 0, 'region_25'),
      Text(21, 0, 'region_10'),
      Text(22, 0, 'region_27'),
      Text(23, 0, 'region_30'),
      Text(24, 0, 'region_12'),
      Text(25, 0, 'region_21'),
      Text(26, 0, 'region_8'),
      Text(27, 0, 'region_32'),
      Text(28, 0, 'region_6'),
      Text(29, 0, 'region_33'),
      Text(30, 0, 'region_24'),
      Text(31, 0, 'region_3'),
      Text(32, 0, 'region_9'),
      Text(33, 0, 'region_18')])




![output_17_1](/images/2023-12-03-LogisticRegression/output_17_1.png)

# Null 처리





```python
hr_df.isna().mean()
```




    department              0.000000
    region                  0.000000
    education               0.043953
    gender                  0.000000
    recruitment_channel     0.000000
    no_of_trainings         0.000000
    age                     0.000000
    previous_year_rating    0.075244
    length_of_service       0.000000
    awards_won?             0.000000
    avg_training_score      0.000000
    is_promoted             0.000000
    dtype: float64




```python
hr_df['education'].value_counts()
```




    Bachelor's          36669
    Master's & above    14925
    Below Secondary       805
    Name: education, dtype: int64




```python
hr_df['education']=hr_df['education'].fillna('unknown')
```


```python
hr_df['previous_year_rating'].value_counts()
```




    3.0    18618
    5.0    11741
    4.0     9877
    1.0     6223
    2.0     4225
    Name: previous_year_rating, dtype: int64




```python
hr_df=hr_df.dropna()
```


```python
hr_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 50684 entries, 0 to 54807
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   department            50684 non-null  object 
     1   region                50684 non-null  object 
     2   education             50684 non-null  object 
     3   gender                50684 non-null  object 
     4   recruitment_channel   50684 non-null  object 
     5   no_of_trainings       50684 non-null  int64  
     6   age                   50684 non-null  int64  
     7   previous_year_rating  50684 non-null  float64
     8   length_of_service     50684 non-null  int64  
     9   awards_won?           50684 non-null  int64  
     10  avg_training_score    50684 non-null  int64  
     11  is_promoted           50684 non-null  int64  
    dtypes: float64(1), int64(6), object(5)
    memory usage: 5.0+ MB



```python
for i in ['department','region','education','gender','recruitment_channel']:
    print(i,hr_df[i].nunique())
```

    department 9
    region 34
    education 4
    gender 2
    recruitment_channel 3



# One-hot Encoding



```python
hr_df=pd.get_dummies(hr_df,columns=['department','region','education','gender','recruitment_channel'])
```


```python
hr_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>no_of_trainings</th>
      <th>age</th>
      <th>previous_year_rating</th>
      <th>length_of_service</th>
      <th>awards_won?</th>
      <th>avg_training_score</th>
      <th>is_promoted</th>
      <th>department_Analytics</th>
      <th>department_Finance</th>
      <th>department_HR</th>
      <th>...</th>
      <th>region_region_9</th>
      <th>education_Bachelor's</th>
      <th>education_Below Secondary</th>
      <th>education_Master's &amp; above</th>
      <th>education_unknown</th>
      <th>gender_f</th>
      <th>gender_m</th>
      <th>recruitment_channel_other</th>
      <th>recruitment_channel_referred</th>
      <th>recruitment_channel_sourcing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>35</td>
      <td>5.0</td>
      <td>8</td>
      <td>0</td>
      <td>49</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>30</td>
      <td>5.0</td>
      <td>4</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>34</td>
      <td>3.0</td>
      <td>7</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 59 columns</p>
</div>



# Train- Test 분리




```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(hr_df.drop('is_promoted',axis=1),hr_df['is_promoted'],test_size=0.3,random_state=10)
```


```python
from sklearn.linear_model import LogisticRegression
```


```python
lr = LogisticRegression()
```


```python
lr.fit(X_train,y_train)
```

    C:\Users\MINUK\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(



# LogisticRegression



    LogisticRegression()




```python
pred = lr.predict(X_test)
```

* [도큐먼트]
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regression#sklearn.l[…]egression
* 독립 변수의 선형 결합을 이용하여 사건의 발생 가능성을 예측하는데 사용되는 통계 기법
* 로지스틱 회귀, 서포트 벡터 머신과 같은 알고리즘은 이진 분류만 가능(2개의 클래스 판별만 가능)
* 3개 이상의 클래스에 대한 판별을 진행하는 경우 아래와 같은 전략으로 판별
    1. one-vs-rest(OvR): K개의 클래스가 존재할 때, 1개의 클래스를 제외한 다른 클래스를 K개 만들어, 각가의 이진 분류에 대한 확률을 구하고, 총합을 통해 최종 클래스를 판별
    2. one-vs-one(OvO): 4개의 계정을 구분하는 클래스가 존재한다고 할 때, 0vs1, 0vs2, 0vs3... 2vs3까지의 NX(N-1) /2 개의 분류기를 만들어 가장 많이 샹성으로 선택된 클래스를 판별

대부분 OvsR 전략을 선호


```python
from sklearn.metrics import accuracy_score, confusion_matrix
```


```python
accuracy_score(y_test, pred) #타켓값의 분포가 한쪽으로 치우쳐저 있기 때문에 의미가 크게 없음 (대부분 승진 못함)
```




    0.9250953570958832




```python
hr_df['is_promoted'].value_counts()
```




    0    46355
    1     4329
    Name: is_promoted, dtype: int64




```python
confusion_matrix(y_test,pred) #오차 행렬: 분류 모델이 정확한지 평가할 때 많이 활용 (TN, FP)(FN, TP)

#               (예측값)
#
# 실제값)     TN         FP
#             FN         TP

```




    array([[13855,   107],
           [ 1032,   212]], dtype=int64)




```python
sns.heatmap(confusion_matrix(y_test,pred),annot=True,cmap='Reds')
```




    <AxesSubplot:>




![output_38_1](/images/2023-12-03-LogisticRegression/output_38_1.png)


### 정밀도(precision)
* TP / (TP+FP)
* 무조건 양성으로 판단해서 계산하는 방법

### 재현율(recall)
* TP / (TP+FN)
* 정확하게 감지한 양성 샘플의 비율
* 민감도 또는 TPR(True Positive Rate)라고 부름

### f1 score
* 정밀도와 재현율의 조화 평균을 나타내는 지표
$$2*\frac{정밀도 * 재현율}{정밀도 + 재현율}=\frac{TP}{TP+\frac{FN+FP}{2}}$$

