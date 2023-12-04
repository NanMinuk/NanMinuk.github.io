---
title: RandomForest
tags: Machine_Learnig
typora-root-url: ../
---

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
hotel_df = pd.read_csv('hotel.csv')
```


```python
hotel_df.head()
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status_date</th>
      <th>name</th>
      <th>email</th>
      <th>phone-number</th>
      <th>credit_card</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-07-01</td>
      <td>Ernest Barnes</td>
      <td>Ernest.Barnes31@outlook.com</td>
      <td>669-792-1661</td>
      <td>************4322</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-07-01</td>
      <td>Andrea Baker</td>
      <td>Andrea_Baker94@aol.com</td>
      <td>858-637-6955</td>
      <td>************9157</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-07-02</td>
      <td>Rebecca Parker</td>
      <td>Rebecca_Parker@comcast.net</td>
      <td>652-885-2745</td>
      <td>************3734</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-07-02</td>
      <td>Laura Murray</td>
      <td>Laura_M@gmail.com</td>
      <td>364-656-8427</td>
      <td>************5677</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-07-03</td>
      <td>Linda Hines</td>
      <td>LHines@verizon.com</td>
      <td>713-226-5883</td>
      <td>************5498</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
pd.set_option('display.max_columns', 40)
```


```python
hotel_df.head()
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status_date</th>
      <th>name</th>
      <th>email</th>
      <th>phone-number</th>
      <th>credit_card</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-07-01</td>
      <td>Ernest Barnes</td>
      <td>Ernest.Barnes31@outlook.com</td>
      <td>669-792-1661</td>
      <td>************4322</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-07-01</td>
      <td>Andrea Baker</td>
      <td>Andrea_Baker94@aol.com</td>
      <td>858-637-6955</td>
      <td>************9157</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-07-02</td>
      <td>Rebecca Parker</td>
      <td>Rebecca_Parker@comcast.net</td>
      <td>652-885-2745</td>
      <td>************3734</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>2015-07-02</td>
      <td>Laura Murray</td>
      <td>Laura_M@gmail.com</td>
      <td>364-656-8427</td>
      <td>************5677</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>2015-07-03</td>
      <td>Linda Hines</td>
      <td>LHines@verizon.com</td>
      <td>713-226-5883</td>
      <td>************5498</td>
    </tr>
  </tbody>
</table>
</div>




```python
hotel_df.drop(['arrival_date_week_number', 'arrival_date_day_of_month', 'reservation_status_date', 'name', 'email', 'phone-number', 'credit_card'], axis=1, inplace=True)
```


```python
hotel_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 119390 entries, 0 to 119389
    Data columns (total 25 columns):
     #   Column                          Non-Null Count   Dtype  
    ---  ------                          --------------   -----  
     0   hotel                           119390 non-null  object 
     1   is_canceled                     119390 non-null  int64  
     2   lead_time                       119390 non-null  int64  
     3   arrival_date_year               119390 non-null  int64  
     4   arrival_date_month              119390 non-null  object 
     5   stays_in_weekend_nights         119390 non-null  int64  
     6   stays_in_week_nights            119390 non-null  int64  
     7   adults                          119390 non-null  int64  
     8   children                        119386 non-null  float64
     9   babies                          119390 non-null  int64  
     10  meal                            119390 non-null  object 
     11  country                         118902 non-null  object 
     12  distribution_channel            119390 non-null  object 
     13  is_repeated_guest               119390 non-null  int64  
     14  previous_cancellations          119390 non-null  int64  
     15  previous_bookings_not_canceled  119390 non-null  int64  
     16  reserved_room_type              119390 non-null  object 
     17  assigned_room_type              119390 non-null  object 
     18  booking_changes                 119390 non-null  int64  
     19  deposit_type                    119390 non-null  object 
     20  days_in_waiting_list            119390 non-null  int64  
     21  customer_type                   119390 non-null  object 
     22  adr                             119390 non-null  float64
     23  required_car_parking_spaces     119390 non-null  int64  
     24  total_of_special_requests       119390 non-null  int64  
    dtypes: float64(2), int64(14), object(9)
    memory usage: 22.8+ MB
    


```python
hotel_df.describe()
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
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>booking_changes</th>
      <th>days_in_waiting_list</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119386.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
      <td>119390.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.370416</td>
      <td>104.011416</td>
      <td>2016.156554</td>
      <td>0.927599</td>
      <td>2.500302</td>
      <td>1.856403</td>
      <td>0.103890</td>
      <td>0.007949</td>
      <td>0.031912</td>
      <td>0.087118</td>
      <td>0.137097</td>
      <td>0.221124</td>
      <td>2.321149</td>
      <td>101.831122</td>
      <td>0.062518</td>
      <td>0.571363</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.482918</td>
      <td>106.863097</td>
      <td>0.707476</td>
      <td>0.998613</td>
      <td>1.908286</td>
      <td>0.579261</td>
      <td>0.398561</td>
      <td>0.097436</td>
      <td>0.175767</td>
      <td>0.844336</td>
      <td>1.497437</td>
      <td>0.652306</td>
      <td>17.594721</td>
      <td>50.535790</td>
      <td>0.245291</td>
      <td>0.792798</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2015.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-6.380000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>2016.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>69.290000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>69.000000</td>
      <td>2016.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>94.575000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>160.000000</td>
      <td>2017.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>126.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>737.000000</td>
      <td>2017.000000</td>
      <td>19.000000</td>
      <td>50.000000</td>
      <td>55.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>26.000000</td>
      <td>72.000000</td>
      <td>21.000000</td>
      <td>391.000000</td>
      <td>5400.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.displot(hotel_df['lead_time'])
```




    <seaborn.axisgrid.FacetGrid at 0x1de4dd16ca0>




![png](output_8_1.png)



```python
sns.boxplot(y=hotel_df['lead_time'])
```




    <AxesSubplot:ylabel='lead_time'>




![png](output_9_1.png)



```python
hotel_df[hotel_df['adr'] < 0] # 취소가 되지 않았는데 요금이 마이너스임
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14969</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>195</td>
      <td>2017</td>
      <td>March</td>
      <td>4</td>
      <td>6</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>H</td>
      <td>2</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>-6.38</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
hotel_df = hotel_df[hotel_df['adr'] >= 0]
```


```python
sns.barplot(x=hotel_df['distribution_channel'], y=hotel_df['is_canceled'])
```




    <AxesSubplot:xlabel='distribution_channel', ylabel='is_canceled'>




![png](output_12_1.png)



```python
hotel_df['distribution_channel'].value_counts()
```




    TA/TO        97870
    Direct       14644
    Corporate     6677
    GDS            193
    Undefined        5
    Name: distribution_channel, dtype: int64




```python
sns.barplot(x=hotel_df['hotel'], y=hotel_df['is_canceled'])
```




    <AxesSubplot:xlabel='hotel', ylabel='is_canceled'>




![png](output_14_1.png)



```python
sns.barplot(x=hotel_df['arrival_date_year'], y=hotel_df['is_canceled'])
```




    <AxesSubplot:xlabel='arrival_date_year', ylabel='is_canceled'>




![png](output_15_1.png)



```python
plt.figure(figsize=(15, 5))
sns.barplot(x=hotel_df['arrival_date_month'], y=hotel_df['is_canceled'])
```




    <AxesSubplot:xlabel='arrival_date_month', ylabel='is_canceled'>




![png](output_16_1.png)



```python
# sns.barplot(x=hotel_df['arrival_date_month'], y=hotel_df['is_canceled'], order=['January', 'February'..])

import calendar

print(calendar.month_name[1])
print(calendar.month_name[2])
```

    January
    February
    


```python
months = []

for i in range(1, 13):
    months.append(calendar.month_name[i])
```


```python
months
```




    ['January',
     'February',
     'March',
     'April',
     'May',
     'June',
     'July',
     'August',
     'September',
     'October',
     'November',
     'December']




```python
plt.figure(figsize=(15, 5))
sns.barplot(x=hotel_df['arrival_date_month'], y=hotel_df['is_canceled'], order=months)
```




    <AxesSubplot:xlabel='arrival_date_month', ylabel='is_canceled'>




![png](output_20_1.png)



```python
sns.barplot(x=hotel_df['is_repeated_guest'], y=hotel_df['is_canceled'])
```




    <AxesSubplot:xlabel='is_repeated_guest', ylabel='is_canceled'>




![png](output_21_1.png)



```python
sns.barplot(x=hotel_df['deposit_type'], y=hotel_df['is_canceled'])
```




    <AxesSubplot:xlabel='deposit_type', ylabel='is_canceled'>




![png](output_22_1.png)



```python
hotel_df['deposit_type'].value_counts()
```




    No Deposit    104640
    Non Refund     14587
    Refundable       162
    Name: deposit_type, dtype: int64




```python
plt.figure(figsize=(20, 20))
sns.heatmap(hotel_df.corr(), cmap='coolwarm', vmax=1, vmin=-1, annot=True)
```




    <AxesSubplot:>




![png](output_24_1.png)



```python
hotel_df.isna().mean()
```




    hotel                             0.000000
    is_canceled                       0.000000
    lead_time                         0.000000
    arrival_date_year                 0.000000
    arrival_date_month                0.000000
    stays_in_weekend_nights           0.000000
    stays_in_week_nights              0.000000
    adults                            0.000000
    children                          0.000034
    babies                            0.000000
    meal                              0.000000
    country                           0.004087
    distribution_channel              0.000000
    is_repeated_guest                 0.000000
    previous_cancellations            0.000000
    previous_bookings_not_canceled    0.000000
    reserved_room_type                0.000000
    assigned_room_type                0.000000
    booking_changes                   0.000000
    deposit_type                      0.000000
    days_in_waiting_list              0.000000
    customer_type                     0.000000
    adr                               0.000000
    required_car_parking_spaces       0.000000
    total_of_special_requests         0.000000
    dtype: float64




```python
hotel_df = hotel_df.dropna()
```


```python
hotel_df.head()
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
hotel_df[hotel_df['adults'] == 0]
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2224</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>1</td>
      <td>2015</td>
      <td>October</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>I</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2409</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>October</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>I</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3181</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>36</td>
      <td>2015</td>
      <td>November</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>ESP</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>165</td>
      <td>2015</td>
      <td>December</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>PRT</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>122</td>
      <td>Transient-Party</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3708</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>165</td>
      <td>2015</td>
      <td>December</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>PRT</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>122</td>
      <td>Transient-Party</td>
      <td>0.00</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>117204</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>296</td>
      <td>2017</td>
      <td>July</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>B</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.85</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>117274</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>276</td>
      <td>2017</td>
      <td>July</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>BB</td>
      <td>DEU</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>B</td>
      <td>B</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>93.64</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>117303</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>291</td>
      <td>2017</td>
      <td>July</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>B</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.85</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>117453</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>159</td>
      <td>2017</td>
      <td>July</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>SC</td>
      <td>FRA</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>121.88</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>118200</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>10</td>
      <td>2017</td>
      <td>August</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>3.0</td>
      <td>0</td>
      <td>BB</td>
      <td>MAR</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>B</td>
      <td>A</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>6.00</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>393 rows × 25 columns</p>
</div>




```python
hotel_df['people'] = hotel_df['adults'] + hotel_df['children'] + hotel_df['babies']
```


```python
hotel_df[hotel_df['people'] == 0]
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>people</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2224</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>1</td>
      <td>2015</td>
      <td>October</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>I</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2409</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>October</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>I</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3181</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>36</td>
      <td>2015</td>
      <td>November</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>ESP</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>165</td>
      <td>2015</td>
      <td>December</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>PRT</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>122</td>
      <td>Transient-Party</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3708</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>165</td>
      <td>2015</td>
      <td>December</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>PRT</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>122</td>
      <td>Transient-Party</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115029</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>107</td>
      <td>2017</td>
      <td>June</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>CHE</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>100.80</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>115091</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>1</td>
      <td>2017</td>
      <td>June</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>E</td>
      <td>K</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>116251</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>44</td>
      <td>2017</td>
      <td>July</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>SWE</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>K</td>
      <td>2</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>73.80</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>116534</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>2</td>
      <td>2017</td>
      <td>July</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>RUS</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>K</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>22.86</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>117087</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>170</td>
      <td>2017</td>
      <td>July</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>BRA</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>170 rows × 26 columns</p>
</div>




```python
hotel_df = hotel_df[hotel_df['people'] != 0]
```


```python
hotel_df.head()
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>people</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
hotel_df['total_nights'] = hotel_df['stays_in_weekend_nights'] + hotel_df['stays_in_week_nights']
```


```python
hotel_df[hotel_df['total_nights'] == 0]
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>people</th>
      <th>total_nights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>111</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>H</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>2</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>168</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>E</td>
      <td>H</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>196</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>8</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115483</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>15</td>
      <td>2017</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>SC</td>
      <td>FRA</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117701</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>0</td>
      <td>2017</td>
      <td>August</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>TA/TO</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>118029</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>0</td>
      <td>2017</td>
      <td>August</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>118631</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>78</td>
      <td>2017</td>
      <td>August</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>K</td>
      <td>7</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>118963</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>1</td>
      <td>2017</td>
      <td>August</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>640 rows × 27 columns</p>
</div>




```python
hotel_df['arrival_date_month'].apply(lambda x: 'spring' if x in ['March', 'April', 'May'] else 'summer' if x in ['June', 'July', 'August'] else 'fall' if x in ['September', 'October', 'November'] else 'winter')
```




    0         summer
    1         summer
    2         summer
    3         summer
    4         summer
               ...  
    119385    summer
    119386    summer
    119387    summer
    119388    summer
    119389    summer
    Name: arrival_date_month, Length: 118727, dtype: object




```python
season_dic = {'spring':[3, 4, 5], 'summer':[6, 7, 8], 'fall':[9, 10, 11], 'winter':[12, 1, 2]}
```


```python
new_season_dic = {}

for i in season_dic:
    for j in season_dic[i]:
        new_season_dic[calendar.month_name[j]] = i
```


```python
new_season_dic
```




    {'March': 'spring',
     'April': 'spring',
     'May': 'spring',
     'June': 'summer',
     'July': 'summer',
     'August': 'summer',
     'September': 'fall',
     'October': 'fall',
     'November': 'fall',
     'December': 'winter',
     'January': 'winter',
     'February': 'winter'}




```python
hotel_df['season'] = hotel_df['arrival_date_month'].map(new_season_dic)
```


```python
hotel_df.head()
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>people</th>
      <th>total_nights</th>
      <th>season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>summer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>summer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>2</td>
      <td>summer</td>
    </tr>
  </tbody>
</table>
</div>




```python
hotel_df['expected_room_type'] = (hotel_df['reserved_room_type'] == hotel_df['assigned_room_type']).astype(int)
```


```python
hotel_df.head()
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>people</th>
      <th>total_nights</th>
      <th>season</th>
      <th>expected_room_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>2</td>
      <td>summer</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
hotel_df['cancel_rate'] = hotel_df['previous_cancellations'] / (hotel_df['previous_cancellations'] + hotel_df['previous_bookings_not_canceled'])
```


```python
hotel_df.head()
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>people</th>
      <th>total_nights</th>
      <th>season</th>
      <th>expected_room_type</th>
      <th>cancel_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>2</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
hotel_df[hotel_df['cancel_rate'].isna()]
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>people</th>
      <th>total_nights</th>
      <th>season</th>
      <th>expected_room_type</th>
      <th>cancel_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>C</td>
      <td>C</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>C</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.00</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.00</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.00</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>2</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
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
      <th>119385</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>23</td>
      <td>2017</td>
      <td>August</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>BEL</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>96.14</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>7</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>119386</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>102</td>
      <td>2017</td>
      <td>August</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>FRA</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>E</td>
      <td>E</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>225.43</td>
      <td>0</td>
      <td>2</td>
      <td>3.0</td>
      <td>7</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>119387</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>34</td>
      <td>2017</td>
      <td>August</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>DEU</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>D</td>
      <td>D</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>157.71</td>
      <td>0</td>
      <td>4</td>
      <td>2.0</td>
      <td>7</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>119388</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>109</td>
      <td>2017</td>
      <td>August</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>GBR</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>104.40</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>7</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>119389</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>205</td>
      <td>2017</td>
      <td>August</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>HB</td>
      <td>DEU</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>151.20</td>
      <td>0</td>
      <td>2</td>
      <td>2.0</td>
      <td>9</td>
      <td>summer</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>109523 rows × 30 columns</p>
</div>




```python
hotel_df[~hotel_df['cancel_rate'].isna()]
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>people</th>
      <th>total_nights</th>
      <th>season</th>
      <th>expected_room_type</th>
      <th>cancel_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13808</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>6</td>
      <td>2016</td>
      <td>January</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>D</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>2</td>
      <td>winter</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13813</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>1</td>
      <td>2016</td>
      <td>February</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>D</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>2</td>
      <td>winter</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13814</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>6</td>
      <td>2016</td>
      <td>November</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>fall</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13815</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>6</td>
      <td>2017</td>
      <td>January</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>winter</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13817</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>1</td>
      <td>2017</td>
      <td>February</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>2</td>
      <td>winter</td>
      <td>1</td>
      <td>0.0</td>
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
      <th>117424</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>3</td>
      <td>2017</td>
      <td>August</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>95.0</td>
      <td>0</td>
      <td>4</td>
      <td>3.0</td>
      <td>1</td>
      <td>summer</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>117841</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2017</td>
      <td>August</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>65.0</td>
      <td>0</td>
      <td>2</td>
      <td>1.0</td>
      <td>2</td>
      <td>summer</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>118581</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>11</td>
      <td>2017</td>
      <td>August</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>FRA</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>D</td>
      <td>D</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Group</td>
      <td>125.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>2</td>
      <td>summer</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>118651</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>189</td>
      <td>2017</td>
      <td>August</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>ITA</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>A</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>119.0</td>
      <td>0</td>
      <td>3</td>
      <td>2.0</td>
      <td>2</td>
      <td>summer</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>118654</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>189</td>
      <td>2017</td>
      <td>August</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>ITA</td>
      <td>TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>A</td>
      <td>1</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>119.0</td>
      <td>0</td>
      <td>2</td>
      <td>2.0</td>
      <td>2</td>
      <td>summer</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>9204 rows × 30 columns</p>
</div>




```python
hotel_df[hotel_df['cancel_rate']>0]
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
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>meal</th>
      <th>country</th>
      <th>distribution_channel</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>reserved_room_type</th>
      <th>assigned_room_type</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>people</th>
      <th>total_nights</th>
      <th>season</th>
      <th>expected_room_type</th>
      <th>cancel_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13825</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>6</td>
      <td>2016</td>
      <td>March</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>spring</td>
      <td>1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>13826</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2016</td>
      <td>June</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
      <td>1</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>13827</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>8</td>
      <td>2016</td>
      <td>September</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>2</td>
      <td>fall</td>
      <td>1</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>13855</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>5</td>
      <td>2015</td>
      <td>November</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>fall</td>
      <td>1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>13856</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>December</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>winter</td>
      <td>1</td>
      <td>0.333333</td>
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
      <th>111356</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>10</td>
      <td>2017</td>
      <td>June</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
      <td>1</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>111357</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>20</td>
      <td>2017</td>
      <td>July</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
      <td>summer</td>
      <td>1</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>111358</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>8</td>
      <td>2017</td>
      <td>July</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>65.0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
      <td>1</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>111359</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2017</td>
      <td>August</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>A</td>
      <td>A</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
      <td>1</td>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>111925</th>
      <td>City Hotel</td>
      <td>1</td>
      <td>6</td>
      <td>2017</td>
      <td>July</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Corporate</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>A</td>
      <td>D</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>65.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>summer</td>
      <td>0</td>
      <td>0.500000</td>
    </tr>
  </tbody>
</table>
<p>6442 rows × 30 columns</p>
</div>




```python
hotel_df['cancel_rate'] = hotel_df['cancel_rate'].fillna(-99)
```


```python
hotel_df['country'].dtype # Object
```




    dtype('O')




```python
hotel_df['adr'].dtype # float64
```




    dtype('float64')




```python
obj_list = []
for i in hotel_df.columns:
    if hotel_df[i].dtype == 'O':
        obj_list.append(i)
```


```python
obj_list
```




    ['hotel',
     'arrival_date_month',
     'meal',
     'country',
     'distribution_channel',
     'reserved_room_type',
     'assigned_room_type',
     'deposit_type',
     'customer_type',
     'season']




```python
for i in obj_list:
    print(i, hotel_df[i].nunique())
```

    hotel 2
    arrival_date_month 12
    meal 5
    country 177
    distribution_channel 5
    reserved_room_type 9
    assigned_room_type 11
    deposit_type 3
    customer_type 4
    season 4
    


```python
hotel_df['meal'].value_counts()
```




    BB           91788
    HB           14429
    SC           10547
    Undefined     1165
    FB             798
    Name: meal, dtype: int64




```python
hotel_df.drop(['country', 'meal'], axis=1, inplace=True)
```


```python
obj_list.remove('country')
obj_list.remove('meal')
```


```python
hotel_df = pd.get_dummies(hotel_df, columns=obj_list)
```


```python
hotel_df.head()
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
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>children</th>
      <th>babies</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>booking_changes</th>
      <th>days_in_waiting_list</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>people</th>
      <th>total_nights</th>
      <th>expected_room_type</th>
      <th>cancel_rate</th>
      <th>...</th>
      <th>assigned_room_type_C</th>
      <th>assigned_room_type_D</th>
      <th>assigned_room_type_E</th>
      <th>assigned_room_type_F</th>
      <th>assigned_room_type_G</th>
      <th>assigned_room_type_H</th>
      <th>assigned_room_type_I</th>
      <th>assigned_room_type_K</th>
      <th>assigned_room_type_L</th>
      <th>deposit_type_No Deposit</th>
      <th>deposit_type_Non Refund</th>
      <th>deposit_type_Refundable</th>
      <th>customer_type_Contract</th>
      <th>customer_type_Group</th>
      <th>customer_type_Transient</th>
      <th>customer_type_Transient-Party</th>
      <th>season_fall</th>
      <th>season_spring</th>
      <th>season_summer</th>
      <th>season_winter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>-99.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>-99.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>-99.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>-99.0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>2</td>
      <td>1</td>
      <td>-99.0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 70 columns</p>
</div>



# 앙상블(Ensemble) 모델
* 머신러닝 앙상블이란 여러개의 머신러닝 모델을 이용해 최적의 답을 찾아내는 기법
    * 보팅(Voting): 투표를 통해 결과를 도출
        * 다른 알고리즘 model을 조합해서 사용
        * 분류를 할 때 voting이라는 하이퍼 파라미터를 사용
            * hard: class를 0, 1로 분류 예측하는 이진 분류일 때 결과 값에 대한 다수 class를 차용
	            * 예) 분류를 예측한 값이 1, 0, 0, 1, 1 이였다면 1이 3표, 0이 2표이므로 1이 최종값으로 예측을 하게 됨 
            * soft: 각각의 확률의 평균 값을 계산한 다음 가장 확률이 높은 값으로 확정
	            * 예) class 0이 나올 확률이 (0.4, 0.9, 0.9, 0.4, 0.4)이었고, class1이 나올 확률이 (0.6, 0.1, 0.1, 0.6, 0.6)이었다면
		            * class0이 나올 최종 확률은 (0.4+0.9+0.9+0.4+0.4)/5 = 0.6
		            * class1이 나올 최종 확률은 (0.6+0.1+0.1+0.6+0.6)/5 = 0.4가 되기 때문에 class0이 최종으로 확률이 높은 것으로 확정
    * 배깅(Bagging): 샘플 중복 생성을 통해 결과를 도출
        * 같은 알고리즘 내에서 다른 sample 조합을 사용
        * Bagging은 Bootstrap Aggregating의 줄임말
        * Bootstrap은 여러개의 dataset을 중첩을 허용하여 샘플링하고 분할하는 방식
	        * 예) 데이터셋의 구성이 [1, 2, 3, 4, 5]로 되어 있다면
		        1. group1 = [1, 2, 3]
		        2. group2 = [1, 3, 4]
		        3. group3 = [2, 3, 5]
        * 대표적인 Bagging 앙상블: Random Forest, Bagging
    * 부스팅(Boosting): 이전 오차를 보완해가면서 가중치를 부여
    * 스태킹(Stacking): 여러 모델을 기반으로 예측된 결과를 통해 meta 모델이 다시 한번 예측

### Random Forest
* Decision Tree 기반 Bagging 앙상블
* 굉장히 인기있는 앙상블 모델
* 사용성이 쉽고, 성능도 꽤 우수한 편


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(hotel_df.drop('is_canceled', axis=1), hotel_df['is_canceled'], test_size=0.4, random_state=10)
```


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rf = RandomForestClassifier()
```


```python
rf.fit(X_train,y_train)
```




    RandomForestClassifier()




```python
pred1 = rf.predict(X_test)
```


```python
proba1 = rf.predict_proba(X_test)
```


```python
proba1
```




    array([[0.04      , 0.96      ],
           [0.19      , 0.81      ],
           [0.904     , 0.096     ],
           ...,
           [0.97133323, 0.02866677],
           [0.16      , 0.84      ],
           [1.        , 0.        ]])




```python
proba1[0]
```




    array([0.04, 0.96])




```python
proba1[:,1]
```




    array([0.96      , 0.81      , 0.096     , ..., 0.02866677, 0.84      ,
           0.        ])




```python
pred1
```




    array([1, 1, 0, ..., 0, 1, 0], dtype=int64)




```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
```


```python
accuracy_score(y_test,pred1)
```




    0.8550041060411446




```python
confusion_matrix(y_test,pred1)
```




    array([[27568,  2321],
           [ 4565, 13037]], dtype=int64)




```python
print(classification_report(y_test,pred1))
```

                  precision    recall  f1-score   support
    
               0       0.86      0.92      0.89     29889
               1       0.85      0.74      0.79     17602
    
        accuracy                           0.86     47491
       macro avg       0.85      0.83      0.84     47491
    weighted avg       0.85      0.86      0.85     47491
    
    

### precision( 정밀도)
* 1이라고 예측한 것 중 얼마 만큼을 제대로 맞췄는가?
* TP / (TP+FP)
* 13025 / 13054 + 2347 = 0.85

### recall(재현율)
* 실제 1인것 중에, 얼마 만큼을 제대로 맞췄는가?
* TP / TP + FN
* 13025 / 13025 + 4577 = 0.74

### f1 score(조화평균)
* 정밀도와 재현율의 조화 평균을 나타내는 지표
$$2*\frac{정밀도 * 재현율}{정밀도 + 재현율}=\frac{TP}{TP+\frac{FN+FP}{2}}$$


```python
roc_auc_score(y_test,proba1[:,1])
```




    0.9240438315096159



### ROC Curve
* 이진 분류의 성능을 측정하는 도구
* FPR(False Positive Rate): FP/TN+FP
    * 거짓 양성 비율(실제값은 음성이지만 양성으로 잘못 분류)
    * FP / TN+FP
* TPR(True Positive Rate): TP/FN+TP
    * 참인 양성 비율(실제로도 양성이고 양성으로 잘 분류)
    * TP / FN+TP

### AUC
* Area Under the ROC Curve의 줄임말
* ROC커브와 직선 사이의 면적을 의미
* AUC값의 범위는 0.5~1이며 값이 클수록 예측의 정확도가 높음


```python
rf2 = RandomForestClassifier(max_depth = 10, random_state=10)
rf2.fit(X_train,y_train)
proba2 = rf2.predict_proba(X_test)
roc_auc_score(y_test,proba2[:,1])
```




    0.8876252067125507




```python
rf2 = RandomForestClassifier(max_depth = 30, random_state=10)
rf2.fit(X_train,y_train)
proba2 = rf2.predict_proba(X_test)
roc_auc_score(y_test,proba2[:,1])
```




    0.9269714667749065




```python
rf2 = RandomForestClassifier(max_depth = 50, random_state=10)
rf2.fit(X_train,y_train)
proba2 = rf2.predict_proba(X_test)
roc_auc_score(y_test,proba2[:,1])
```




    0.9242749559576547




```python
rf2 = RandomForestClassifier(min_samples_split=3,random_state=10)
rf2.fit(X_train,y_train)
proba2 = rf2.predict_proba(X_test)
roc_auc_score(y_test,proba2[:,1])
```




    0.9257678219091355




```python
rf2 = RandomForestClassifier(min_samples_split=5,random_state=10)
rf2.fit(X_train,y_train)
proba2 = rf2.predict_proba(X_test)
roc_auc_score(y_test,proba2[:,1])
```




    0.9260347889319027




```python
rf2 = RandomForestClassifier(min_samples_split=7,random_state=10)
rf2.fit(X_train,y_train)
proba2 = rf2.predict_proba(X_test)
roc_auc_score(y_test,proba2[:,1])
```




    0.9262233278697595




```python
rf2 = RandomForestClassifier(max_depth=30, min_samples_split=7,random_state=10)
rf2.fit(X_train,y_train)
proba2 = rf2.predict_proba(X_test)
roc_auc_score(y_test,proba2[:,1])
```




    0.9266751349572633




```python
from sklearn.model_selection import GridSearchCV
```


```python
params={
    'max_depth':[None,10,30,50],
    'min_samples_split':[2,3,5,7]
}
```


```python
rf3 = RandomForestClassifier(random_state=10)
```


```python
grid_df = GridSearchCV(rf3,params,cv=7)
```


```python
grid_df.fit(X_train,y_train)
```




    GridSearchCV(cv=7, estimator=RandomForestClassifier(random_state=10),
                 param_grid={'max_depth': [None, 10, 30, 50],
                             'min_samples_split': [2, 3, 5, 7]})




```python
grid_df.best_params_
```




    {'max_depth': 30, 'min_samples_split': 2}




```python
proba3=grid_df.predict_proba(X_test)
```


```python
roc_auc_score(y_test,proba3[:,1])
```




    0.9269714667749065



### feature_importance_ (피쳐 중요도)
* 결정트리에서 노드를 분기할 때, 해당 피쳐가 클래스를 나누는데 
얼마나 영향을 미쳤는지를 표기하는 척도
* 0이면 클래스를 구분하는데 피쳐가 선택되지 않았다는 것이며, 1이면 해당 피쳐가 클래스를 완벽하게 나누었다는 것을 의미


```python
rf4=RandomForestClassifier(max_depth=30,min_samples_split=2,random_state=10)
```


```python
rf4.fit(X_train,y_train)
```




    RandomForestClassifier(max_depth=30, random_state=10)




```python
proba4=rf4.predict_proba(X_test)
```


```python
roc_auc_score(y_test,proba3[:,1])
```




    0.9269714667749065




```python
rf4.feature_importances_
```




    array([1.54529670e-01, 2.57855507e-02, 2.56518042e-02, 3.77711699e-02,
           1.10615748e-02, 6.01001771e-03, 9.66409207e-04, 2.12749549e-03,
           2.81271369e-02, 3.09361841e-03, 2.39043101e-02, 3.23038619e-03,
           1.16688211e-01, 2.04288020e-02, 6.29629984e-02, 1.42327830e-02,
           4.16322463e-02, 2.79109937e-02, 3.34810324e-02, 7.80308407e-03,
           7.85960326e-03, 4.84668446e-03, 5.54241539e-03, 3.37386853e-03,
           3.54734471e-03, 2.84774513e-03, 5.49462524e-03, 4.75293925e-03,
           4.26656914e-03, 4.90206654e-03, 3.57269347e-03, 4.22446917e-03,
           3.90074159e-03, 3.34173599e-03, 8.37006579e-03, 2.70318518e-04,
           1.20204365e-02, 1.50064421e-07, 5.93172870e-03, 8.98507590e-04,
           6.99627718e-04, 4.40155187e-03, 2.46450114e-03, 1.25595109e-03,
           1.13289691e-03, 3.92410084e-04, 2.91222977e-05, 9.48913970e-03,
           1.44648022e-03, 1.29775223e-03, 5.73165347e-03, 2.81227177e-03,
           1.53125931e-03, 1.25451978e-03, 4.42639255e-04, 1.30436753e-04,
           1.17928570e-04, 0.00000000e+00, 8.92745542e-02, 8.46475090e-02,
           4.88755276e-04, 2.88359142e-03, 4.19871799e-04, 1.82069535e-02,
           1.13734141e-02, 4.90008877e-03, 5.32195436e-03, 5.77080319e-03,
           4.71835794e-03])




```python
feat_imp = pd.DataFrame({'features': X_train.columns,'importances':rf4.feature_importances_})
```


```python
top10=feat_imp.sort_values('importances',ascending=False).head(10)
```


```python
plt.figure(figsize=(15,10))
sns.barplot(x='importances',y='features',data=top10)
```




    <AxesSubplot:xlabel='importances', ylabel='features'>




![png](output_105_1.png)



```python

```
