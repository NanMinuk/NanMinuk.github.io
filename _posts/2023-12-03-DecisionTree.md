---
title: DecisionTree
tags: Machine_Learnig
typora-root-url: ../
---

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




```python
sns.displot(bike_df['count'])
```




    <seaborn.axisgrid.FacetGrid at 0x22b93614b20>




![png](output_8_1.png)



```python
sns.boxplot(y=bike_df['count'])
```




    <AxesSubplot:ylabel='count'>




![png](output_9_1.png)



```python
sns.scatterplot(x='feels_like', y='count', data=bike_df, alpha=0.2)
```




    <AxesSubplot:xlabel='feels_like', ylabel='count'>




![png](output_10_1.png)



```python
sns.scatterplot(x='pressure', y='count', data=bike_df, alpha=0.2)
```




    <AxesSubplot:xlabel='pressure', ylabel='count'>




![png](output_11_1.png)



```python
sns.scatterplot(x='wind_speed', y='count', data=bike_df, alpha=0.2)
```




    <AxesSubplot:xlabel='wind_speed', ylabel='count'>




![png](output_12_1.png)



```python
sns.scatterplot(x='wind_deg', y='count', data=bike_df, alpha=0.2)
```




    <AxesSubplot:xlabel='wind_deg', ylabel='count'>




![png](output_13_1.png)



```python
plt.figure(figsize=(10, 5))
sns.boxplot(x='weather_main', y='count', data=bike_df,palette='Set3')
```




    <AxesSubplot:xlabel='weather_main', ylabel='count'>




![png](output_14_1.png)



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


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_3060\1504819505.py in <module>
          1 plt.figure(figsize=(15, 5))
    ----> 2 sns.lineplot(x='date', y='count', data=bike_df)
          3 plt.xticks(rotation=45)
          4 plt.show()
    

    ~\anaconda3\lib\site-packages\seaborn\_decorators.py in inner_f(*args, **kwargs)
         44             )
         45         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 46         return f(**kwargs)
         47     return inner_f
         48 
    

    ~\anaconda3\lib\site-packages\seaborn\relational.py in lineplot(x, y, hue, size, style, data, palette, hue_order, hue_norm, sizes, size_order, size_norm, dashes, markers, style_order, units, estimator, ci, n_boot, seed, sort, err_style, err_kws, legend, ax, **kwargs)
        708     p._attach(ax)
        709 
    --> 710     p.plot(ax, kwargs)
        711     return ax
        712 
    

    ~\anaconda3\lib\site-packages\seaborn\relational.py in plot(self, ax, kws)
        497                     err = "estimator must be None when specifying units"
        498                     raise ValueError(err)
    --> 499                 x, y, y_ci = self.aggregate(y, x, u)
        500             else:
        501                 y_ci = None
    

    ~\anaconda3\lib\site-packages\seaborn\relational.py in aggregate(self, vals, grouper, units)
        412                                columns=["low", "high"]).stack()
        413         else:
    --> 414             cis = grouped.apply(bootstrapped_cis)
        415 
        416         # Unpack the CIs into "wide" format for plotting
    

    ~\anaconda3\lib\site-packages\pandas\core\groupby\generic.py in apply(self, func, *args, **kwargs)
        242     )
        243     def apply(self, func, *args, **kwargs):
    --> 244         return super().apply(func, *args, **kwargs)
        245 
        246     @doc(_agg_template, examples=_agg_examples_doc, klass="Series")
    

    ~\anaconda3\lib\site-packages\pandas\core\groupby\groupby.py in apply(self, func, *args, **kwargs)
       1421         with option_context("mode.chained_assignment", None):
       1422             try:
    -> 1423                 result = self._python_apply_general(f, self._selected_obj)
       1424             except TypeError:
       1425                 # gh-20949
    

    ~\anaconda3\lib\site-packages\pandas\core\groupby\groupby.py in _python_apply_general(self, f, data, not_indexed_same)
       1462             data after applying f
       1463         """
    -> 1464         values, mutated = self.grouper.apply(f, data, self.axis)
       1465 
       1466         if not_indexed_same is None:
    

    ~\anaconda3\lib\site-packages\pandas\core\groupby\ops.py in apply(self, f, data, axis)
        759             # group might be modified
        760             group_axes = group.axes
    --> 761             res = f(group)
        762             if not mutated and not _is_indexed_like(res, group_axes, axis):
        763                 mutated = True
    

    ~\anaconda3\lib\site-packages\seaborn\relational.py in bootstrapped_cis(vals)
        393                 return null_ci
        394 
    --> 395             boots = bootstrap(vals, func=func, n_boot=n_boot, seed=seed)
        396             cis = ci_func(boots, ci)
        397             return pd.Series(cis, ["low", "high"])
    

    ~\anaconda3\lib\site-packages\seaborn\algorithms.py in bootstrap(*args, **kwargs)
         82     for i in range(int(n_boot)):
         83         resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
    ---> 84         sample = [a.take(resampler, axis=0) for a in args]
         85         boot_dist.append(f(*sample, **func_kwargs))
         86     return np.array(boot_dist)
    

    ~\anaconda3\lib\site-packages\seaborn\algorithms.py in <listcomp>(.0)
         82     for i in range(int(n_boot)):
         83         resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
    ---> 84         sample = [a.take(resampler, axis=0) for a in args]
         85         boot_dist.append(f(*sample, **func_kwargs))
         86     return np.array(boot_dist)
    

    KeyboardInterrupt: 



![png](output_30_1.png)



```python
bike_df[bike_df['year'] == 2019].groupby('month').mean()['count']
```


```python
# 4월 데이터가 없음
bike_df[bike_df['year'] == 2020].groupby('month').mean()['count']
```


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


```python
bike_df['date'].apply(covid)
```


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


```python
bike_df['day_night'] = bike_df['hour'].apply(lambda x: 'night' if x>=21 else 'late evening' if x>=19 else 'early evening' if x>=17 else 'late afternoon' if x>= 16 else 'early afternoon' if x>=13 else 'late morning' if x>=11 else 'early morning' if x>=5 else 'night')
```


```python
bike_df.head()
```


```python
bike_df.drop(['datetime', 'month', 'date'], axis=1, inplace=True)
```


```python
bike_df.head()
```


```python
for i in ['weather_main', 'hour', 'covid', 'season', 'day_night']:
    print(i, bike_df[i].nunique())
```


```python
bike_df['weather_main'].unique() # 묶을 수 있으면 묶어주는것이 성능상으로 유리할 수 있음
```


```python
pd.get_dummies(bike_df, columns=['weather_main', 'covid', 'season', 'day_night'])
```


```python
bike_df.drop(['hour'], axis=1, inplace=True)
```


```python
bike_df = pd.get_dummies(bike_df, columns=['weather_main', 'covid', 'season', 'day_night'])
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(bike_df.drop('count', axis=1), bike_df['count'], test_size=0.3, random_state=10)
```


```python
from sklearn.tree import DecisionTreeRegressor
```


```python
dt = DecisionTreeRegressor(random_state=10)
```


```python
dt.fit(X_train, y_train)
```


```python
dt.predict(X_test)
```


```python
pred_1 = dt.predict(X_test)
```


```python
sns.scatterplot(y_test, pred_1)
```


```python
from sklearn.metrics import mean_squared_error
```


```python
mean_squared_error(y_test, pred_1, squared=False)
```


```python
from sklearn.linear_model import LinearRegression
```


```python
lr = LinearRegression()
```


```python
lr.fit(X_train, y_train)
```


```python
pred_2 = lr.predict(X_test)
```


```python
sns.scatterplot(y_test, pred_2)
```


```python
mean_squared_error(y_test, pred_2, squared=False)
```


```python
lr.fit(X_train, np.log(y_train))
```


```python
np.log(0)
```


```python
lr.fit(X_train, np.log(y_train+1))
```


```python
pred_3 = lr.predict(X_test)
```


```python
pred_3 = np.exp(pred_3) - 1
```


```python
mean_squared_error(y_test, pred_3,squared=False)
```


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


```python
pred_4 = dt.predict(X_test)
```


```python
mean_squared_error(y_test,pred_4,squared=False)
```


```python
graph_tree(dt)
```


```python
from sklearn.tree import plot_tree
```


```python
plt.figure(figsize=(25,15))
plot_tree(dt,max_depth =3, fontsize =13)
plt.show()
```


```python
X_train
```


```python
plt.figure(figsize=(25,15))
plot_tree(dt,max_depth =3, fontsize =13, feature_names=X_train.columns)
plt.show()
```
