---
title: DecisionTree
tags: Machine_Learning
typora-root-url: ../
---

# 데이터로드

사용 데이터셋: [Bike Sharing Demand | Kaggle](https://www.kaggle.com/c/bike-sharing-demand/data)



```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.show()
```


```python
bike_df = pd.read_csv('bike.csv')
```


```python
bike_df.head()
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
      <th>datetime</th>
      <th>count</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>temp</th>
      <th>feels_like</th>
      <th>temp_min</th>
      <th>temp_max</th>
      <th>pressure</th>
      <th>humidity</th>
      <th>wind_speed</th>
      <th>wind_deg</th>
      <th>rain_1h</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
      <th>weather_main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01 0:00</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>-7.17</td>
      <td>-12.73</td>
      <td>-8.56</td>
      <td>-7.09</td>
      <td>1030</td>
      <td>53</td>
      <td>3.6</td>
      <td>310</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20</td>
      <td>Clouds</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-01 1:00</td>
      <td>49</td>
      <td>1</td>
      <td>0</td>
      <td>-7.35</td>
      <td>-13.81</td>
      <td>-9.03</td>
      <td>-7.15</td>
      <td>1030</td>
      <td>49</td>
      <td>4.6</td>
      <td>310</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>Clear</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-01 2:00</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>-7.88</td>
      <td>-14.05</td>
      <td>-9.03</td>
      <td>-7.69</td>
      <td>1031</td>
      <td>52</td>
      <td>4.1</td>
      <td>310</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>Clear</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-01 3:00</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>-8.10</td>
      <td>-14.32</td>
      <td>-9.36</td>
      <td>-7.89</td>
      <td>1031</td>
      <td>49</td>
      <td>4.1</td>
      <td>310</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>Clear</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-01 4:00</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>-8.19</td>
      <td>-14.43</td>
      <td>-9.46</td>
      <td>-8.09</td>
      <td>1031</td>
      <td>49</td>
      <td>4.1</td>
      <td>330</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>Clear</td>
    </tr>
  </tbody>
</table>
</div>




```python
bike_df['workingday'] == 1
```




    0        False
    1        False
    2        False
    3        False
    4        False
             ...  
    33374     True
    33375     True
    33376     True
    33377     True
    33378     True
    Name: workingday, Length: 33379, dtype: bool




```python
bike_df['holiday'] == 0
```




    0        False
    1        False
    2        False
    3        False
    4        False
             ...  
    33374     True
    33375     True
    33376     True
    33377     True
    33378     True
    Name: holiday, Length: 33379, dtype: bool




```python
bike_df[bike_df['workingday'] == 0]['holiday'].unique()
```




    array([1, 0], dtype=int64)




```python
bike_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 33379 entries, 0 to 33378
    Data columns (total 16 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   datetime      33379 non-null  object 
     1   count         33379 non-null  int64  
     2   holiday       33379 non-null  int64  
     3   workingday    33379 non-null  int64  
     4   temp          33379 non-null  float64
     5   feels_like    33379 non-null  float64
     6   temp_min      33379 non-null  float64
     7   temp_max      33379 non-null  float64
     8   pressure      33379 non-null  int64  
     9   humidity      33379 non-null  int64  
     10  wind_speed    33379 non-null  float64
     11  wind_deg      33379 non-null  int64  
     12  rain_1h       6771 non-null   float64
     13  snow_1h       326 non-null    float64
     14  clouds_all    33379 non-null  int64  
     15  weather_main  33379 non-null  object 
    dtypes: float64(7), int64(7), object(2)
    memory usage: 4.1+ MB



```python
bike_df.describe()
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
      <th>count</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>temp</th>
      <th>feels_like</th>
      <th>temp_min</th>
      <th>temp_max</th>
      <th>pressure</th>
      <th>humidity</th>
      <th>wind_speed</th>
      <th>wind_deg</th>
      <th>rain_1h</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>33379.000000</td>
      <td>33379.000000</td>
      <td>33379.000000</td>
      <td>33379.000000</td>
      <td>33379.000000</td>
      <td>33379.000000</td>
      <td>33379.000000</td>
      <td>33379.000000</td>
      <td>33379.000000</td>
      <td>33379.000000</td>
      <td>33379.000000</td>
      <td>6771.000000</td>
      <td>326.000000</td>
      <td>33379.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>333.139788</td>
      <td>0.030618</td>
      <td>0.681327</td>
      <td>15.213087</td>
      <td>14.994843</td>
      <td>13.532648</td>
      <td>16.105542</td>
      <td>1017.071602</td>
      <td>67.818628</td>
      <td>1.829340</td>
      <td>174.022919</td>
      <td>1.216475</td>
      <td>0.641380</td>
      <td>63.213997</td>
    </tr>
    <tr>
      <th>std</th>
      <td>336.519514</td>
      <td>0.172283</td>
      <td>0.465969</td>
      <td>9.908964</td>
      <td>11.176487</td>
      <td>9.993094</td>
      <td>9.984839</td>
      <td>7.379420</td>
      <td>18.422105</td>
      <td>1.703747</td>
      <td>113.844334</td>
      <td>2.056222</td>
      <td>0.571087</td>
      <td>30.825936</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-12.790000</td>
      <td>-18.910000</td>
      <td>-15.140000</td>
      <td>-12.290000</td>
      <td>980.000000</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>59.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.860000</td>
      <td>5.880000</td>
      <td>5.230000</td>
      <td>7.730000</td>
      <td>1012.000000</td>
      <td>53.000000</td>
      <td>0.450000</td>
      <td>62.000000</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>236.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>15.650000</td>
      <td>15.020000</td>
      <td>13.910000</td>
      <td>16.590000</td>
      <td>1017.000000</td>
      <td>70.000000</td>
      <td>1.340000</td>
      <td>180.000000</td>
      <td>0.530000</td>
      <td>0.420000</td>
      <td>75.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>495.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>23.800000</td>
      <td>24.140000</td>
      <td>21.970000</td>
      <td>24.390000</td>
      <td>1022.000000</td>
      <td>84.000000</td>
      <td>2.600000</td>
      <td>285.000000</td>
      <td>1.300000</td>
      <td>1.000000</td>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2038.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>36.710000</td>
      <td>43.710000</td>
      <td>35.380000</td>
      <td>38.810000</td>
      <td>1044.000000</td>
      <td>100.000000</td>
      <td>16.980000</td>
      <td>360.000000</td>
      <td>54.050000</td>
      <td>3.300000</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>



# 데이터 시각화


```python
sns.distplot(bike_df['count'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x194ff25b308>




![output_11_1](/images/2023-12-03-DecisionTree/output_11_1-1701699986012-1.png)



```python
sns.boxplot(y=bike_df['count'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x19481d5f6c8>




![output_12_1](/images/2023-12-03-DecisionTree/output_12_1-1701699990807-3.png)



```python
sns.scatterplot(x='feels_like', y='count', data=bike_df, alpha=0.2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x194fb4dcf88>




![output_13_1](/images/2023-12-03-DecisionTree/output_13_1-1701699994793-5.png)



```python
sns.scatterplot(x='pressure', y='count', data=bike_df, alpha=0.2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x194fd0e4348>




![output_14_1](/images/2023-12-03-DecisionTree/output_14_1-1701699998411-7.png)



```python
sns.scatterplot(x='wind_speed', y='count', data=bike_df, alpha=0.2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x194fd182b88>




![output_15_1](/images/2023-12-03-DecisionTree/output_15_1.png)



```python
sns.scatterplot(x='wind_deg', y='count', data=bike_df, alpha=0.2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x194fd1a9c48>

![output_16_1](/images/2023-12-03-DecisionTree/output_16_1.png)



```python
plt.figure(figsize=(10, 5))
sns.boxplot(x='weather_main', y='count', data=bike_df,palette='Set3')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x194fd1dcf08>




!![output_17_1](/images/2023-12-03-DecisionTree/output_17_1.png)


# Null값 처리


```python
bike_df.isna().sum()
```




    datetime            0
    count               0
    holiday             0
    workingday          0
    temp                0
    feels_like          0
    temp_min            0
    temp_max            0
    pressure            0
    humidity            0
    wind_speed          0
    wind_deg            0
    rain_1h         26608
    snow_1h         33053
    clouds_all          0
    weather_main        0
    dtype: int64




```python
bike_df.isna().mean()
```




    datetime        0.000000
    count           0.000000
    holiday         0.000000
    workingday      0.000000
    temp            0.000000
    feels_like      0.000000
    temp_min        0.000000
    temp_max        0.000000
    pressure        0.000000
    humidity        0.000000
    wind_speed      0.000000
    wind_deg        0.000000
    rain_1h         0.797148
    snow_1h         0.990233
    clouds_all      0.000000
    weather_main    0.000000
    dtype: float64




```python
# bike_df['rain_1h] = bike_df['rain_1h'].fillna(0)
# bike_df['snow_1h] = bike_df['snow_1h'].fillna(0)
bike_df = bike_df.fillna(0)
```


```python
bike_df.head()
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
      <th>datetime</th>
      <th>count</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>temp</th>
      <th>feels_like</th>
      <th>temp_min</th>
      <th>temp_max</th>
      <th>pressure</th>
      <th>humidity</th>
      <th>wind_speed</th>
      <th>wind_deg</th>
      <th>rain_1h</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
      <th>weather_main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01 0:00</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>-7.17</td>
      <td>-12.73</td>
      <td>-8.56</td>
      <td>-7.09</td>
      <td>1030</td>
      <td>53</td>
      <td>3.6</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20</td>
      <td>Clouds</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-01 1:00</td>
      <td>49</td>
      <td>1</td>
      <td>0</td>
      <td>-7.35</td>
      <td>-13.81</td>
      <td>-9.03</td>
      <td>-7.15</td>
      <td>1030</td>
      <td>49</td>
      <td>4.6</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-01 2:00</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>-7.88</td>
      <td>-14.05</td>
      <td>-9.03</td>
      <td>-7.69</td>
      <td>1031</td>
      <td>52</td>
      <td>4.1</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-01 3:00</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>-8.10</td>
      <td>-14.32</td>
      <td>-9.36</td>
      <td>-7.89</td>
      <td>1031</td>
      <td>49</td>
      <td>4.1</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-01 4:00</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>-8.19</td>
      <td>-14.43</td>
      <td>-9.46</td>
      <td>-8.09</td>
      <td>1031</td>
      <td>49</td>
      <td>4.1</td>
      <td>330</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
    </tr>
  </tbody>
</table>
</div>




```python
bike_df.isna().mean()
```




    datetime        0.0
    count           0.0
    holiday         0.0
    workingday      0.0
    temp            0.0
    feels_like      0.0
    temp_min        0.0
    temp_max        0.0
    pressure        0.0
    humidity        0.0
    wind_speed      0.0
    wind_deg        0.0
    rain_1h         0.0
    snow_1h         0.0
    clouds_all      0.0
    weather_main    0.0
    dtype: float64




```python
bike_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 33379 entries, 0 to 33378
    Data columns (total 16 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   datetime      33379 non-null  object 
     1   count         33379 non-null  int64  
     2   holiday       33379 non-null  int64  
     3   workingday    33379 non-null  int64  
     4   temp          33379 non-null  float64
     5   feels_like    33379 non-null  float64
     6   temp_min      33379 non-null  float64
     7   temp_max      33379 non-null  float64
     8   pressure      33379 non-null  int64  
     9   humidity      33379 non-null  int64  
     10  wind_speed    33379 non-null  float64
     11  wind_deg      33379 non-null  int64  
     12  rain_1h       33379 non-null  float64
     13  snow_1h       33379 non-null  float64
     14  clouds_all    33379 non-null  int64  
     15  weather_main  33379 non-null  object 
    dtypes: float64(7), int64(7), object(2)
    memory usage: 4.1+ MB



```python
bike_df['datetime'] = pd.to_datetime(bike_df['datetime'])
```


```python
bike_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 33379 entries, 0 to 33378
    Data columns (total 16 columns):
     #   Column        Non-Null Count  Dtype         
    ---  ------        --------------  -----         
     0   datetime      33379 non-null  datetime64[ns]
     1   count         33379 non-null  int64         
     2   holiday       33379 non-null  int64         
     3   workingday    33379 non-null  int64         
     4   temp          33379 non-null  float64       
     5   feels_like    33379 non-null  float64       
     6   temp_min      33379 non-null  float64       
     7   temp_max      33379 non-null  float64       
     8   pressure      33379 non-null  int64         
     9   humidity      33379 non-null  int64         
     10  wind_speed    33379 non-null  float64       
     11  wind_deg      33379 non-null  int64         
     12  rain_1h       33379 non-null  float64       
     13  snow_1h       33379 non-null  float64       
     14  clouds_all    33379 non-null  int64         
     15  weather_main  33379 non-null  object        
    dtypes: datetime64[ns](1), float64(7), int64(7), object(1)
    memory usage: 4.1+ MB



```python
bike_df.head()
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
      <th>datetime</th>
      <th>count</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>temp</th>
      <th>feels_like</th>
      <th>temp_min</th>
      <th>temp_max</th>
      <th>pressure</th>
      <th>humidity</th>
      <th>wind_speed</th>
      <th>wind_deg</th>
      <th>rain_1h</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
      <th>weather_main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01 00:00:00</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>-7.17</td>
      <td>-12.73</td>
      <td>-8.56</td>
      <td>-7.09</td>
      <td>1030</td>
      <td>53</td>
      <td>3.6</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20</td>
      <td>Clouds</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-01 01:00:00</td>
      <td>49</td>
      <td>1</td>
      <td>0</td>
      <td>-7.35</td>
      <td>-13.81</td>
      <td>-9.03</td>
      <td>-7.15</td>
      <td>1030</td>
      <td>49</td>
      <td>4.6</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-01 02:00:00</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>-7.88</td>
      <td>-14.05</td>
      <td>-9.03</td>
      <td>-7.69</td>
      <td>1031</td>
      <td>52</td>
      <td>4.1</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-01 03:00:00</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>-8.10</td>
      <td>-14.32</td>
      <td>-9.36</td>
      <td>-7.89</td>
      <td>1031</td>
      <td>49</td>
      <td>4.1</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-01 04:00:00</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>-8.19</td>
      <td>-14.43</td>
      <td>-9.46</td>
      <td>-8.09</td>
      <td>1031</td>
      <td>49</td>
      <td>4.1</td>
      <td>330</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
    </tr>
  </tbody>
</table>
</div>




```python
bike_df['year'] = bike_df['datetime'].dt.year # 해
```


```python
bike_df['month'] = bike_df['datetime'].dt.month # 월, 계절적인 요인
```


```python
bike_df['hour'] = bike_df['datetime'].dt.hour # 낮, 밤
```


```python
bike_df.head()
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
      <th>datetime</th>
      <th>count</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>temp</th>
      <th>feels_like</th>
      <th>temp_min</th>
      <th>temp_max</th>
      <th>pressure</th>
      <th>humidity</th>
      <th>wind_speed</th>
      <th>wind_deg</th>
      <th>rain_1h</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
      <th>weather_main</th>
      <th>year</th>
      <th>month</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01 00:00:00</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>-7.17</td>
      <td>-12.73</td>
      <td>-8.56</td>
      <td>-7.09</td>
      <td>1030</td>
      <td>53</td>
      <td>3.6</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20</td>
      <td>Clouds</td>
      <td>2018</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-01 01:00:00</td>
      <td>49</td>
      <td>1</td>
      <td>0</td>
      <td>-7.35</td>
      <td>-13.81</td>
      <td>-9.03</td>
      <td>-7.15</td>
      <td>1030</td>
      <td>49</td>
      <td>4.6</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-01 02:00:00</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>-7.88</td>
      <td>-14.05</td>
      <td>-9.03</td>
      <td>-7.69</td>
      <td>1031</td>
      <td>52</td>
      <td>4.1</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-01 03:00:00</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>-8.10</td>
      <td>-14.32</td>
      <td>-9.36</td>
      <td>-7.89</td>
      <td>1031</td>
      <td>49</td>
      <td>4.1</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-01 04:00:00</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>-8.19</td>
      <td>-14.43</td>
      <td>-9.46</td>
      <td>-8.09</td>
      <td>1031</td>
      <td>49</td>
      <td>4.1</td>
      <td>330</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
bike_df['datetime'].dt.date
```




    0        2018-01-01
    1        2018-01-01
    2        2018-01-01
    3        2018-01-01
    4        2018-01-01
                ...    
    33374    2021-08-31
    33375    2021-08-31
    33376    2021-08-31
    33377    2021-08-31
    33378    2021-08-31
    Name: datetime, Length: 33379, dtype: object




```python
bike_df['date'] = bike_df['datetime'].dt.date
```


```python
plt.figure(figsize=(15, 5))
sns.lineplot(x='date', y='count', data=bike_df)
plt.xticks(rotation=45)
plt.show()
```




![output_34_0](/images/2023-12-03-DecisionTree/output_34_0.png)

```python
bike_df[bike_df['year'] == 2019].groupby('month').mean()['count']
```




    month
    1     193.368862
    2     221.857718
    3     326.564456
    4     482.931694
    5     438.027848
    6     478.480053
    7     472.745785
    8     481.267366
    9     500.862069
    10    446.279070
    11    307.295393
    12    213.148886
    Name: count, dtype: float64




```python
# 4월 데이터가 없음
bike_df[bike_df['year'] == 2020].groupby('month').mean()['count']
```




    month
    1     260.445997
    2     255.894320
    3     217.135241
    5     196.581064
    6     290.900937
    7     299.811688
    8     331.528809
    9     338.876478
    10    293.640777
    11    240.507324
    12    138.993540
    Name: count, dtype: float64



# 코로나 이전 이후 분리


```python
# covid
# 2020-04-01 이전 : precovid
# 2021-04-01 이전 : covid
# 이후 : postcovid
def covid(date):
    if date < '2020-04-01':
        return 'precovid'
    elif date < '2021-04-01':
        return 'covid'
    else:
        return 'postcovid'
```


```python
#bike_df['date'] < '2020-04-01'
```


```python
#covid(bike_df['date'])
```


```python
def covid(date):
    if str(date) < '2020-04-01':
        return 'precovid'
    elif str(date) < '2021-04-01':
        return 'covid'
    else:
        return 'postcovid'
```


```python
covid(bike_df['date'])
```




    'precovid'




```python
bike_df['date'].apply(covid)
```




    0         precovid
    1         precovid
    2         precovid
    3         precovid
    4         precovid
               ...    
    33374    postcovid
    33375    postcovid
    33376    postcovid
    33377    postcovid
    33378    postcovid
    Name: date, Length: 33379, dtype: object




```python
bike_df['covid'] = bike_df['date'].apply(lambda date: 'precovid' if str(date) < '2020-04-01' else 'covid' if str(date) < '2021-04-01' else 'postcovid')
```


```python
# season
# 3월~5월 : spring
# 6월~8월 : summer
# 9월~11월 : fall
# 나머지: winter
bike_df['season'] = bike_df['month'].apply(lambda x: 'winter' if x==12 else 'fall' if x>=9 else 'summer' if x>=6 else 'spring' if x>=3 else 'winter')
```


```python
bike_df[['month', 'season']]
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
      <th>month</th>
      <th>season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>winter</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>winter</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>winter</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>winter</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>winter</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>33374</th>
      <td>8</td>
      <td>summer</td>
    </tr>
    <tr>
      <th>33375</th>
      <td>8</td>
      <td>summer</td>
    </tr>
    <tr>
      <th>33376</th>
      <td>8</td>
      <td>summer</td>
    </tr>
    <tr>
      <th>33377</th>
      <td>8</td>
      <td>summer</td>
    </tr>
    <tr>
      <th>33378</th>
      <td>8</td>
      <td>summer</td>
    </tr>
  </tbody>
</table>
<p>33379 rows × 2 columns</p>
</div>




```python
bike_df['day_night'] = bike_df['hour'].apply(lambda x: 'night' if x>=21 else 'late evening' if x>=19 else 'early evening' if x>=17 else 'late afternoon' if x>= 16 else 'early afternoon' if x>=13 else 'late morning' if x>=11 else 'early morning' if x>=5 else 'night')
```


```python
bike_df.head()
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
      <th>datetime</th>
      <th>count</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>temp</th>
      <th>feels_like</th>
      <th>temp_min</th>
      <th>temp_max</th>
      <th>pressure</th>
      <th>humidity</th>
      <th>...</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
      <th>weather_main</th>
      <th>year</th>
      <th>month</th>
      <th>hour</th>
      <th>date</th>
      <th>covid</th>
      <th>season</th>
      <th>day_night</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01 00:00:00</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>-7.17</td>
      <td>-12.73</td>
      <td>-8.56</td>
      <td>-7.09</td>
      <td>1030</td>
      <td>53</td>
      <td>...</td>
      <td>0.0</td>
      <td>20</td>
      <td>Clouds</td>
      <td>2018</td>
      <td>1</td>
      <td>0</td>
      <td>2018-01-01</td>
      <td>precovid</td>
      <td>winter</td>
      <td>night</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-01 01:00:00</td>
      <td>49</td>
      <td>1</td>
      <td>0</td>
      <td>-7.35</td>
      <td>-13.81</td>
      <td>-9.03</td>
      <td>-7.15</td>
      <td>1030</td>
      <td>49</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>2018-01-01</td>
      <td>precovid</td>
      <td>winter</td>
      <td>night</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-01 02:00:00</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>-7.88</td>
      <td>-14.05</td>
      <td>-9.03</td>
      <td>-7.69</td>
      <td>1031</td>
      <td>52</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>2018-01-01</td>
      <td>precovid</td>
      <td>winter</td>
      <td>night</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-01 03:00:00</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>-8.10</td>
      <td>-14.32</td>
      <td>-9.36</td>
      <td>-7.89</td>
      <td>1031</td>
      <td>49</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>2018-01-01</td>
      <td>precovid</td>
      <td>winter</td>
      <td>night</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-01 04:00:00</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>-8.19</td>
      <td>-14.43</td>
      <td>-9.46</td>
      <td>-8.09</td>
      <td>1031</td>
      <td>49</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>2018-01-01</td>
      <td>precovid</td>
      <td>winter</td>
      <td>night</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
bike_df.drop(['datetime', 'month', 'date'], axis=1, inplace=True)
```


```python
bike_df.head()
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
      <th>count</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>temp</th>
      <th>feels_like</th>
      <th>temp_min</th>
      <th>temp_max</th>
      <th>pressure</th>
      <th>humidity</th>
      <th>wind_speed</th>
      <th>wind_deg</th>
      <th>rain_1h</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
      <th>weather_main</th>
      <th>year</th>
      <th>hour</th>
      <th>covid</th>
      <th>season</th>
      <th>day_night</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>-7.17</td>
      <td>-12.73</td>
      <td>-8.56</td>
      <td>-7.09</td>
      <td>1030</td>
      <td>53</td>
      <td>3.6</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20</td>
      <td>Clouds</td>
      <td>2018</td>
      <td>0</td>
      <td>precovid</td>
      <td>winter</td>
      <td>night</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>1</td>
      <td>0</td>
      <td>-7.35</td>
      <td>-13.81</td>
      <td>-9.03</td>
      <td>-7.15</td>
      <td>1030</td>
      <td>49</td>
      <td>4.6</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>2018</td>
      <td>1</td>
      <td>precovid</td>
      <td>winter</td>
      <td>night</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>-7.88</td>
      <td>-14.05</td>
      <td>-9.03</td>
      <td>-7.69</td>
      <td>1031</td>
      <td>52</td>
      <td>4.1</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>2018</td>
      <td>2</td>
      <td>precovid</td>
      <td>winter</td>
      <td>night</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>-8.10</td>
      <td>-14.32</td>
      <td>-9.36</td>
      <td>-7.89</td>
      <td>1031</td>
      <td>49</td>
      <td>4.1</td>
      <td>310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>2018</td>
      <td>3</td>
      <td>precovid</td>
      <td>winter</td>
      <td>night</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>-8.19</td>
      <td>-14.43</td>
      <td>-9.46</td>
      <td>-8.09</td>
      <td>1031</td>
      <td>49</td>
      <td>4.1</td>
      <td>330</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>2018</td>
      <td>4</td>
      <td>precovid</td>
      <td>winter</td>
      <td>night</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in ['weather_main', 'hour', 'covid', 'season', 'day_night']:
    print(i, bike_df[i].nunique())
```

    weather_main 11
    hour 24
    covid 3
    season 4
    day_night 7



```python
bike_df['weather_main'].unique() # 묶을 수 있으면 묶어주는것이 성능상으로 유리할 수 있음
```




    array(['Clouds', 'Clear', 'Snow', 'Mist', 'Rain', 'Fog', 'Drizzle',
           'Haze', 'Thunderstorm', 'Smoke', 'Squall'], dtype=object)




```python
pd.get_dummies(bike_df, columns=['weather_main', 'covid', 'season', 'day_night'])
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
      <th>count</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>temp</th>
      <th>feels_like</th>
      <th>temp_min</th>
      <th>temp_max</th>
      <th>pressure</th>
      <th>humidity</th>
      <th>wind_speed</th>
      <th>...</th>
      <th>season_spring</th>
      <th>season_summer</th>
      <th>season_winter</th>
      <th>day_night_early afternoon</th>
      <th>day_night_early evening</th>
      <th>day_night_early morning</th>
      <th>day_night_late afternoon</th>
      <th>day_night_late evening</th>
      <th>day_night_late morning</th>
      <th>day_night_night</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>-7.17</td>
      <td>-12.73</td>
      <td>-8.56</td>
      <td>-7.09</td>
      <td>1030</td>
      <td>53</td>
      <td>3.60</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>1</td>
      <td>0</td>
      <td>-7.35</td>
      <td>-13.81</td>
      <td>-9.03</td>
      <td>-7.15</td>
      <td>1030</td>
      <td>49</td>
      <td>4.60</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>-7.88</td>
      <td>-14.05</td>
      <td>-9.03</td>
      <td>-7.69</td>
      <td>1031</td>
      <td>52</td>
      <td>4.10</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>-8.10</td>
      <td>-14.32</td>
      <td>-9.36</td>
      <td>-7.89</td>
      <td>1031</td>
      <td>49</td>
      <td>4.10</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>-8.19</td>
      <td>-14.43</td>
      <td>-9.46</td>
      <td>-8.09</td>
      <td>1031</td>
      <td>49</td>
      <td>4.10</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>33374</th>
      <td>659</td>
      <td>0</td>
      <td>1</td>
      <td>28.78</td>
      <td>32.79</td>
      <td>26.78</td>
      <td>29.94</td>
      <td>1007</td>
      <td>73</td>
      <td>0.45</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33375</th>
      <td>404</td>
      <td>0</td>
      <td>1</td>
      <td>28.52</td>
      <td>32.37</td>
      <td>26.34</td>
      <td>29.84</td>
      <td>1007</td>
      <td>74</td>
      <td>0.45</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33376</th>
      <td>259</td>
      <td>0</td>
      <td>1</td>
      <td>28.22</td>
      <td>31.85</td>
      <td>26.78</td>
      <td>29.25</td>
      <td>1007</td>
      <td>75</td>
      <td>0.45</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33377</th>
      <td>192</td>
      <td>0</td>
      <td>1</td>
      <td>27.51</td>
      <td>30.42</td>
      <td>26.43</td>
      <td>28.85</td>
      <td>1004</td>
      <td>76</td>
      <td>2.06</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33378</th>
      <td>139</td>
      <td>0</td>
      <td>1</td>
      <td>24.48</td>
      <td>24.73</td>
      <td>23.06</td>
      <td>27.85</td>
      <td>1009</td>
      <td>67</td>
      <td>1.54</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>33379 rows × 41 columns</p>
</div>




```python
bike_df.drop(['hour'], axis=1, inplace=True)
```


```python
bike_df = pd.get_dummies(bike_df, columns=['weather_main', 'covid', 'season', 'day_night'])
```

# Train - Test 분리


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(bike_df.drop('count', axis=1), bike_df['count'], test_size=0.3, random_state=10)
```

# DecisionTree


```python
from sklearn.tree import DecisionTreeRegressor
```


```python
dt = DecisionTreeRegressor(random_state=10)
```


```python
dt.fit(X_train, y_train)
```




    DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=None,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, presort='deprecated',
                          random_state=10, splitter='best')




```python
dt.predict(X_test)
```




    array([255., 471., 487., ...,  55., 705., 868.])




```python
pred_1 = dt.predict(X_test)
```

# 예측 결과


```python
sns.scatterplot(y_test, pred_1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x194fe082c08>




![output_66_1](/images/2023-12-03-DecisionTree/output_66_1.png)



```python
from sklearn.metrics import mean_squared_error
```


```python
mean_squared_error(y_test, pred_1, squared=False)
```




    224.10943208047624




```python
from sklearn.linear_model import LinearRegression
```


```python
lr = LinearRegression()
```


```python
lr.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
pred_2 = lr.predict(X_test)
```


```python
sns.scatterplot(y_test, pred_2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1948071ed48>




![output_73_1](/images/2023-12-03-DecisionTree/output_73_1.png)



```python
mean_squared_error(y_test, pred_2, squared=False)
```




    226.75284145044625



# DecisionTree 시각화


```python
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
```


```python
def graph_tree(model):
    export_graphviz(model, out_file='tree.dot')
    call(['dot', '-Tpng', 'tree.dot', '-o', 'decision-tree.png', '-Gdpi=600'])
    return Image(filename='decision-tree.png', width=1200)
```


```python
dt = DecisionTreeRegressor(random_state=10,max_depth=50, min_samples_leaf=30)
```


```python
dt.fit(X_train,y_train)
```




    DecisionTreeRegressor(ccp_alpha=0.0, criterion='mse', max_depth=50,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=30, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, presort='deprecated',
                          random_state=10, splitter='best')




```python
pred_4 = dt.predict(X_test)
```


```python
mean_squared_error(y_test,pred_4,squared=False)
```




    187.73089614540783




```python
graph_tree(dt)
```




![output_82_0](/images/2023-12-03-DecisionTree/output_82_0.png)




```python
from sklearn.tree import plot_tree
```


```python
plt.figure(figsize=(25,15))
plot_tree(dt,max_depth =3, fontsize =13)
plt.show()
```




![output_84_0](/images/2023-12-03-DecisionTree/output_84_0.png)

```python
X_train
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
      <th>holiday</th>
      <th>workingday</th>
      <th>temp</th>
      <th>feels_like</th>
      <th>temp_min</th>
      <th>temp_max</th>
      <th>pressure</th>
      <th>humidity</th>
      <th>wind_speed</th>
      <th>wind_deg</th>
      <th>...</th>
      <th>season_spring</th>
      <th>season_summer</th>
      <th>season_winter</th>
      <th>day_night_early afternoon</th>
      <th>day_night_early evening</th>
      <th>day_night_early morning</th>
      <th>day_night_late afternoon</th>
      <th>day_night_late evening</th>
      <th>day_night_late morning</th>
      <th>day_night_night</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7300</th>
      <td>0</td>
      <td>1</td>
      <td>22.77</td>
      <td>23.40</td>
      <td>20.97</td>
      <td>22.85</td>
      <td>1000</td>
      <td>88</td>
      <td>9.26</td>
      <td>360</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17642</th>
      <td>0</td>
      <td>1</td>
      <td>10.87</td>
      <td>9.37</td>
      <td>10.00</td>
      <td>11.62</td>
      <td>1019</td>
      <td>52</td>
      <td>0.89</td>
      <td>160</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17027</th>
      <td>0</td>
      <td>1</td>
      <td>20.94</td>
      <td>20.63</td>
      <td>18.71</td>
      <td>22.21</td>
      <td>1012</td>
      <td>59</td>
      <td>1.50</td>
      <td>280</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16529</th>
      <td>0</td>
      <td>1</td>
      <td>16.05</td>
      <td>15.85</td>
      <td>14.47</td>
      <td>16.85</td>
      <td>1021</td>
      <td>82</td>
      <td>1.34</td>
      <td>23</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26007</th>
      <td>0</td>
      <td>1</td>
      <td>12.62</td>
      <td>11.92</td>
      <td>11.60</td>
      <td>13.28</td>
      <td>1015</td>
      <td>76</td>
      <td>0.45</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10201</th>
      <td>0</td>
      <td>1</td>
      <td>-0.37</td>
      <td>-0.37</td>
      <td>-3.26</td>
      <td>0.36</td>
      <td>1016</td>
      <td>47</td>
      <td>1.00</td>
      <td>130</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9372</th>
      <td>0</td>
      <td>0</td>
      <td>8.82</td>
      <td>8.26</td>
      <td>6.58</td>
      <td>11.58</td>
      <td>1014</td>
      <td>91</td>
      <td>1.54</td>
      <td>220</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28017</th>
      <td>0</td>
      <td>0</td>
      <td>0.47</td>
      <td>0.47</td>
      <td>-1.06</td>
      <td>1.62</td>
      <td>1019</td>
      <td>89</td>
      <td>0.45</td>
      <td>84</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29199</th>
      <td>0</td>
      <td>1</td>
      <td>4.76</td>
      <td>4.76</td>
      <td>3.97</td>
      <td>5.51</td>
      <td>1025</td>
      <td>42</td>
      <td>0.45</td>
      <td>172</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17673</th>
      <td>0</td>
      <td>0</td>
      <td>5.11</td>
      <td>2.23</td>
      <td>3.34</td>
      <td>5.92</td>
      <td>1017</td>
      <td>62</td>
      <td>3.60</td>
      <td>330</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>23365 rows × 39 columns</p>
</div>




```python
plt.figure(figsize=(25,15))
plot_tree(dt,max_depth =3, fontsize =13, feature_names=X_train.columns)
plt.show()
```


![output_86_0](/images/2023-12-03-DecisionTree/output_86_0.png)



```python

```
