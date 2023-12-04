---
title: LinearRegression
tags: Machine_Learnig
typora-root-url: ../
---

```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
```


```python
sns.set()
plt.show()
```


```python
rent_df = pd.read_csv('rent.csv')
```


```python
rent_df.head()
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
      <th>Posted On</th>
      <th>BHK</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Floor</th>
      <th>Area Type</th>
      <th>Area Locality</th>
      <th>City</th>
      <th>Furnishing Status</th>
      <th>Tenant Preferred</th>
      <th>Bathroom</th>
      <th>Point of Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-05-18</td>
      <td>2.0</td>
      <td>10000</td>
      <td>1100.0</td>
      <td>Ground out of 2</td>
      <td>Super Area</td>
      <td>Bandel</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-13</td>
      <td>2.0</td>
      <td>20000</td>
      <td>800.0</td>
      <td>1 out of 3</td>
      <td>Super Area</td>
      <td>Phool Bagan, Kankurgachi</td>
      <td>Kolkata</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-05-16</td>
      <td>2.0</td>
      <td>17000</td>
      <td>1000.0</td>
      <td>1 out of 3</td>
      <td>Super Area</td>
      <td>Salt Lake City Sector 2</td>
      <td>Kolkata</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-07-04</td>
      <td>NaN</td>
      <td>10000</td>
      <td>800.0</td>
      <td>1 out of 2</td>
      <td>Super Area</td>
      <td>Dumdum Park</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-05-09</td>
      <td>2.0</td>
      <td>7500</td>
      <td>850.0</td>
      <td>1 out of 2</td>
      <td>Carpet Area</td>
      <td>South Dum Dum</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
  </tbody>
</table>
</div>



* Posted on : 부동산 매물 등록 날짜
* BHK: 배드,홀,키친의 갯수
* rent : 렌트비
* Size : 집 크기
* Floor : 총 층수의 몇층
* Area Type : 사이즈 기준 , 공용공간을 포함하는지, 집만 포함된 공간
* Area Locality: 지역
* City : 지역
* Furnishing Status ; 풀옵션 여부
* Tenant PReferred : 선호하는 가족 형태
* Bathroom : 화장실 갯수
* Point of Contact : 연락할 곳


```python
rent_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4746 entries, 0 to 4745
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Posted On          4746 non-null   object 
     1   BHK                4743 non-null   float64
     2   Rent               4746 non-null   int64  
     3   Size               4741 non-null   float64
     4   Floor              4746 non-null   object 
     5   Area Type          4746 non-null   object 
     6   Area Locality      4746 non-null   object 
     7   City               4746 non-null   object 
     8   Furnishing Status  4746 non-null   object 
     9   Tenant Preferred   4746 non-null   object 
     10  Bathroom           4746 non-null   int64  
     11  Point of Contact   4746 non-null   object 
    dtypes: float64(2), int64(2), object(8)
    memory usage: 445.1+ KB
    


```python
#데이터형에서 숫자만 뽑아서 여러가지 통계값 알려줌
rent_df.describe()
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
      <th>BHK</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Bathroom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4743.000000</td>
      <td>4.746000e+03</td>
      <td>4741.000000</td>
      <td>4746.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.083913</td>
      <td>3.499345e+04</td>
      <td>967.477536</td>
      <td>1.965866</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.832516</td>
      <td>7.810641e+04</td>
      <td>634.532781</td>
      <td>0.884532</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.200000e+03</td>
      <td>10.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>1.000000e+04</td>
      <td>550.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>1.600000e+04</td>
      <td>850.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>3.300000e+04</td>
      <td>1200.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.000000</td>
      <td>3.500000e+06</td>
      <td>8000.000000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
4.746000e+03 # 소수점이 길경우 표시방법
```




    4746.0




```python
4.746000e-03
```




    0.004746




```python
#소수점 둘째자리까지 표시
round(rent_df.describe(),2)
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
      <th>BHK</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Bathroom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4743.00</td>
      <td>4746.00</td>
      <td>4741.00</td>
      <td>4746.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.08</td>
      <td>34993.45</td>
      <td>967.48</td>
      <td>1.97</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.83</td>
      <td>78106.41</td>
      <td>634.53</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00</td>
      <td>1200.00</td>
      <td>10.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.00</td>
      <td>10000.00</td>
      <td>550.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.00</td>
      <td>16000.00</td>
      <td>850.00</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.00</td>
      <td>33000.00</td>
      <td>1200.00</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.00</td>
      <td>3500000.00</td>
      <td>8000.00</td>
      <td>10.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.displot(rent_df['BHK'])
```




    <seaborn.axisgrid.FacetGrid at 0x2b2db9da6d0>




![png](output_10_1.png)



```python
sns.displot(rent_df['Rent'])
```




    <seaborn.axisgrid.FacetGrid at 0x2b2db9c3760>




![png](output_11_1.png)



```python
rent_df['BHK']
```




    0       2.0
    1       2.0
    2       2.0
    3       NaN
    4       2.0
           ... 
    4741    2.0
    4742    3.0
    4743    3.0
    4744    3.0
    4745    2.0
    Name: BHK, Length: 4746, dtype: float64




```python
sns.displot(rent_df['Rent'])
```




    <seaborn.axisgrid.FacetGrid at 0x2b2dc2960a0>




![png](output_13_1.png)



```python
rent_df['Rent'].sort_values() #350000은 아웃라이얼 가능성이 큼(잘못된 데이터 혹은 튀는 데이터)
```




    4076       1200
    285        1500
    471        1800
    2475       2000
    146        2200
             ...   
    1459     700000
    1329     850000
    827     1000000
    1001    1200000
    1837    3500000
    Name: Rent, Length: 4746, dtype: int64




```python
sns.displot(rent_df.drop(1837)['Rent'])
```




    <seaborn.axisgrid.FacetGrid at 0x2b2dba915b0>




![png](output_15_1.png)



```python
sns.displot(rent_df['Size'])
```




    <seaborn.axisgrid.FacetGrid at 0x2b2de88cb80>




![png](output_16_1.png)



```python
sns.boxplot(y=rent_df['Size'])
```




    <AxesSubplot:ylabel='Size'>




![png](output_17_1.png)



```python
sns.boxplot(y=rent_df['BHK'])
```




    <AxesSubplot:ylabel='BHK'>




![png](output_18_1.png)



```python
sns.boxplot(y=rent_df['Rent'])
```




    <AxesSubplot:ylabel='Rent'>




![png](output_19_1.png)



```python
rent_df.isna()
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
      <th>Posted On</th>
      <th>BHK</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Floor</th>
      <th>Area Type</th>
      <th>Area Locality</th>
      <th>City</th>
      <th>Furnishing Status</th>
      <th>Tenant Preferred</th>
      <th>Bathroom</th>
      <th>Point of Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
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
    </tr>
    <tr>
      <th>4741</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4742</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4743</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4744</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4745</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>4746 rows × 12 columns</p>
</div>




```python
rent_df.isna().sum()
```




    Posted On            0
    BHK                  3
    Rent                 0
    Size                 5
    Floor                0
    Area Type            0
    Area Locality        0
    City                 0
    Furnishing Status    0
    Tenant Preferred     0
    Bathroom             0
    Point of Contact     0
    dtype: int64




```python
# Size에 있는 결측치만 지우기
rent_df.dropna(subset=['Size']) # 행 단위로 지움
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
      <th>Posted On</th>
      <th>BHK</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Floor</th>
      <th>Area Type</th>
      <th>Area Locality</th>
      <th>City</th>
      <th>Furnishing Status</th>
      <th>Tenant Preferred</th>
      <th>Bathroom</th>
      <th>Point of Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-05-18</td>
      <td>2.0</td>
      <td>10000</td>
      <td>1100.0</td>
      <td>Ground out of 2</td>
      <td>Super Area</td>
      <td>Bandel</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-13</td>
      <td>2.0</td>
      <td>20000</td>
      <td>800.0</td>
      <td>1 out of 3</td>
      <td>Super Area</td>
      <td>Phool Bagan, Kankurgachi</td>
      <td>Kolkata</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-05-16</td>
      <td>2.0</td>
      <td>17000</td>
      <td>1000.0</td>
      <td>1 out of 3</td>
      <td>Super Area</td>
      <td>Salt Lake City Sector 2</td>
      <td>Kolkata</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-07-04</td>
      <td>NaN</td>
      <td>10000</td>
      <td>800.0</td>
      <td>1 out of 2</td>
      <td>Super Area</td>
      <td>Dumdum Park</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-05-09</td>
      <td>2.0</td>
      <td>7500</td>
      <td>850.0</td>
      <td>1 out of 2</td>
      <td>Carpet Area</td>
      <td>South Dum Dum</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors</td>
      <td>1</td>
      <td>Contact Owner</td>
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
    </tr>
    <tr>
      <th>4741</th>
      <td>2022-05-18</td>
      <td>2.0</td>
      <td>15000</td>
      <td>1000.0</td>
      <td>3 out of 5</td>
      <td>Carpet Area</td>
      <td>Bandam Kommu</td>
      <td>Hyderabad</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4742</th>
      <td>2022-05-15</td>
      <td>3.0</td>
      <td>29000</td>
      <td>2000.0</td>
      <td>1 out of 4</td>
      <td>Super Area</td>
      <td>Manikonda, Hyderabad</td>
      <td>Hyderabad</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>3</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4743</th>
      <td>2022-07-10</td>
      <td>3.0</td>
      <td>35000</td>
      <td>1750.0</td>
      <td>3 out of 5</td>
      <td>Carpet Area</td>
      <td>Himayath Nagar, NH 7</td>
      <td>Hyderabad</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>3</td>
      <td>Contact Agent</td>
    </tr>
    <tr>
      <th>4744</th>
      <td>2022-07-06</td>
      <td>3.0</td>
      <td>45000</td>
      <td>1500.0</td>
      <td>23 out of 34</td>
      <td>Carpet Area</td>
      <td>Gachibowli</td>
      <td>Hyderabad</td>
      <td>Semi-Furnished</td>
      <td>Family</td>
      <td>2</td>
      <td>Contact Agent</td>
    </tr>
    <tr>
      <th>4745</th>
      <td>2022-05-04</td>
      <td>2.0</td>
      <td>15000</td>
      <td>1000.0</td>
      <td>4 out of 5</td>
      <td>Carpet Area</td>
      <td>Suchitra Circle</td>
      <td>Hyderabad</td>
      <td>Unfurnished</td>
      <td>Bachelors</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
  </tbody>
</table>
<p>4741 rows × 12 columns</p>
</div>




```python
# 열단위로 지움
rent_df.dropna(1) # na가 있는 열을 모두삭제
```

    C:\Users\MINUK\AppData\Local\Temp\ipykernel_11324\622210487.py:2: FutureWarning: In a future version of pandas all arguments of DataFrame.dropna will be keyword-only.
      rent_df.dropna(1) # na가 있는 열을 모두삭제
    




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
      <th>Posted On</th>
      <th>Rent</th>
      <th>Floor</th>
      <th>Area Type</th>
      <th>Area Locality</th>
      <th>City</th>
      <th>Furnishing Status</th>
      <th>Tenant Preferred</th>
      <th>Bathroom</th>
      <th>Point of Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-05-18</td>
      <td>10000</td>
      <td>Ground out of 2</td>
      <td>Super Area</td>
      <td>Bandel</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-13</td>
      <td>20000</td>
      <td>1 out of 3</td>
      <td>Super Area</td>
      <td>Phool Bagan, Kankurgachi</td>
      <td>Kolkata</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-05-16</td>
      <td>17000</td>
      <td>1 out of 3</td>
      <td>Super Area</td>
      <td>Salt Lake City Sector 2</td>
      <td>Kolkata</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-07-04</td>
      <td>10000</td>
      <td>1 out of 2</td>
      <td>Super Area</td>
      <td>Dumdum Park</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-05-09</td>
      <td>7500</td>
      <td>1 out of 2</td>
      <td>Carpet Area</td>
      <td>South Dum Dum</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors</td>
      <td>1</td>
      <td>Contact Owner</td>
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
    </tr>
    <tr>
      <th>4741</th>
      <td>2022-05-18</td>
      <td>15000</td>
      <td>3 out of 5</td>
      <td>Carpet Area</td>
      <td>Bandam Kommu</td>
      <td>Hyderabad</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4742</th>
      <td>2022-05-15</td>
      <td>29000</td>
      <td>1 out of 4</td>
      <td>Super Area</td>
      <td>Manikonda, Hyderabad</td>
      <td>Hyderabad</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>3</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4743</th>
      <td>2022-07-10</td>
      <td>35000</td>
      <td>3 out of 5</td>
      <td>Carpet Area</td>
      <td>Himayath Nagar, NH 7</td>
      <td>Hyderabad</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>3</td>
      <td>Contact Agent</td>
    </tr>
    <tr>
      <th>4744</th>
      <td>2022-07-06</td>
      <td>45000</td>
      <td>23 out of 34</td>
      <td>Carpet Area</td>
      <td>Gachibowli</td>
      <td>Hyderabad</td>
      <td>Semi-Furnished</td>
      <td>Family</td>
      <td>2</td>
      <td>Contact Agent</td>
    </tr>
    <tr>
      <th>4745</th>
      <td>2022-05-04</td>
      <td>15000</td>
      <td>4 out of 5</td>
      <td>Carpet Area</td>
      <td>Suchitra Circle</td>
      <td>Hyderabad</td>
      <td>Unfurnished</td>
      <td>Bachelors</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
  </tbody>
</table>
<p>4746 rows × 10 columns</p>
</div>




```python
rent_df.drop('BHK',axis =1)
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
      <th>Posted On</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Floor</th>
      <th>Area Type</th>
      <th>Area Locality</th>
      <th>City</th>
      <th>Furnishing Status</th>
      <th>Tenant Preferred</th>
      <th>Bathroom</th>
      <th>Point of Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-05-18</td>
      <td>10000</td>
      <td>1100.0</td>
      <td>Ground out of 2</td>
      <td>Super Area</td>
      <td>Bandel</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-13</td>
      <td>20000</td>
      <td>800.0</td>
      <td>1 out of 3</td>
      <td>Super Area</td>
      <td>Phool Bagan, Kankurgachi</td>
      <td>Kolkata</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-05-16</td>
      <td>17000</td>
      <td>1000.0</td>
      <td>1 out of 3</td>
      <td>Super Area</td>
      <td>Salt Lake City Sector 2</td>
      <td>Kolkata</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-07-04</td>
      <td>10000</td>
      <td>800.0</td>
      <td>1 out of 2</td>
      <td>Super Area</td>
      <td>Dumdum Park</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-05-09</td>
      <td>7500</td>
      <td>850.0</td>
      <td>1 out of 2</td>
      <td>Carpet Area</td>
      <td>South Dum Dum</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors</td>
      <td>1</td>
      <td>Contact Owner</td>
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
    </tr>
    <tr>
      <th>4741</th>
      <td>2022-05-18</td>
      <td>15000</td>
      <td>1000.0</td>
      <td>3 out of 5</td>
      <td>Carpet Area</td>
      <td>Bandam Kommu</td>
      <td>Hyderabad</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4742</th>
      <td>2022-05-15</td>
      <td>29000</td>
      <td>2000.0</td>
      <td>1 out of 4</td>
      <td>Super Area</td>
      <td>Manikonda, Hyderabad</td>
      <td>Hyderabad</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>3</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4743</th>
      <td>2022-07-10</td>
      <td>35000</td>
      <td>1750.0</td>
      <td>3 out of 5</td>
      <td>Carpet Area</td>
      <td>Himayath Nagar, NH 7</td>
      <td>Hyderabad</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>3</td>
      <td>Contact Agent</td>
    </tr>
    <tr>
      <th>4744</th>
      <td>2022-07-06</td>
      <td>45000</td>
      <td>1500.0</td>
      <td>23 out of 34</td>
      <td>Carpet Area</td>
      <td>Gachibowli</td>
      <td>Hyderabad</td>
      <td>Semi-Furnished</td>
      <td>Family</td>
      <td>2</td>
      <td>Contact Agent</td>
    </tr>
    <tr>
      <th>4745</th>
      <td>2022-05-04</td>
      <td>15000</td>
      <td>1000.0</td>
      <td>4 out of 5</td>
      <td>Carpet Area</td>
      <td>Suchitra Circle</td>
      <td>Hyderabad</td>
      <td>Unfurnished</td>
      <td>Bachelors</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
  </tbody>
</table>
<p>4746 rows × 11 columns</p>
</div>




```python
rent_df[rent_df['Size'].isna()]
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
      <th>Posted On</th>
      <th>BHK</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Floor</th>
      <th>Area Type</th>
      <th>Area Locality</th>
      <th>City</th>
      <th>Furnishing Status</th>
      <th>Tenant Preferred</th>
      <th>Bathroom</th>
      <th>Point of Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>425</th>
      <td>2022-05-22</td>
      <td>2.0</td>
      <td>9000</td>
      <td>NaN</td>
      <td>2 out of 3</td>
      <td>Super Area</td>
      <td>Airport Area Behala</td>
      <td>Kolkata</td>
      <td>Furnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>430</th>
      <td>2022-05-08</td>
      <td>2.0</td>
      <td>8500</td>
      <td>NaN</td>
      <td>Ground out of 1</td>
      <td>Carpet Area</td>
      <td>Nayabad</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4703</th>
      <td>2022-07-06</td>
      <td>2.0</td>
      <td>12000</td>
      <td>NaN</td>
      <td>4 out of 4</td>
      <td>Super Area</td>
      <td>Anandbagh, Secunderabad, Moula Ali Road</td>
      <td>Hyderabad</td>
      <td>Furnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4731</th>
      <td>2022-06-24</td>
      <td>2.0</td>
      <td>13000</td>
      <td>NaN</td>
      <td>2 out of 2</td>
      <td>Super Area</td>
      <td>Manikonda, Outer Ring Road</td>
      <td>Hyderabad</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4732</th>
      <td>2022-07-08</td>
      <td>2.0</td>
      <td>7000</td>
      <td>NaN</td>
      <td>Ground out of 2</td>
      <td>Super Area</td>
      <td>Vinayaka Nagar</td>
      <td>Hyderabad</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
  </tbody>
</table>
</div>




```python
rent_df[rent_df['Size'].isna()].index
```




    Int64Index([425, 430, 4703, 4731, 4732], dtype='int64')




```python
na_index=rent_df[rent_df['Size'].isna()].index
```


```python
na_index
```




    Int64Index([425, 430, 4703, 4731, 4732], dtype='int64')




```python
rent_df.iloc[na_index]
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
      <th>Posted On</th>
      <th>BHK</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Floor</th>
      <th>Area Type</th>
      <th>Area Locality</th>
      <th>City</th>
      <th>Furnishing Status</th>
      <th>Tenant Preferred</th>
      <th>Bathroom</th>
      <th>Point of Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>425</th>
      <td>2022-05-22</td>
      <td>2.0</td>
      <td>9000</td>
      <td>NaN</td>
      <td>2 out of 3</td>
      <td>Super Area</td>
      <td>Airport Area Behala</td>
      <td>Kolkata</td>
      <td>Furnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>430</th>
      <td>2022-05-08</td>
      <td>2.0</td>
      <td>8500</td>
      <td>NaN</td>
      <td>Ground out of 1</td>
      <td>Carpet Area</td>
      <td>Nayabad</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4703</th>
      <td>2022-07-06</td>
      <td>2.0</td>
      <td>12000</td>
      <td>NaN</td>
      <td>4 out of 4</td>
      <td>Super Area</td>
      <td>Anandbagh, Secunderabad, Moula Ali Road</td>
      <td>Hyderabad</td>
      <td>Furnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4731</th>
      <td>2022-06-24</td>
      <td>2.0</td>
      <td>13000</td>
      <td>NaN</td>
      <td>2 out of 2</td>
      <td>Super Area</td>
      <td>Manikonda, Outer Ring Road</td>
      <td>Hyderabad</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4732</th>
      <td>2022-07-08</td>
      <td>2.0</td>
      <td>7000</td>
      <td>NaN</td>
      <td>Ground out of 2</td>
      <td>Super Area</td>
      <td>Vinayaka Nagar</td>
      <td>Hyderabad</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
  </tbody>
</table>
</div>




```python
# boxplot를 확인 후 mean보다는 median을 쓰는게 좋다고 결정!
rent_df.fillna(rent_df.median()).loc[na_index]
```

    C:\Users\MINUK\AppData\Local\Temp\ipykernel_11324\3610673898.py:2: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      rent_df.fillna(rent_df.median()).loc[na_index]
    




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
      <th>Posted On</th>
      <th>BHK</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Floor</th>
      <th>Area Type</th>
      <th>Area Locality</th>
      <th>City</th>
      <th>Furnishing Status</th>
      <th>Tenant Preferred</th>
      <th>Bathroom</th>
      <th>Point of Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>425</th>
      <td>2022-05-22</td>
      <td>2.0</td>
      <td>9000</td>
      <td>850.0</td>
      <td>2 out of 3</td>
      <td>Super Area</td>
      <td>Airport Area Behala</td>
      <td>Kolkata</td>
      <td>Furnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>430</th>
      <td>2022-05-08</td>
      <td>2.0</td>
      <td>8500</td>
      <td>850.0</td>
      <td>Ground out of 1</td>
      <td>Carpet Area</td>
      <td>Nayabad</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4703</th>
      <td>2022-07-06</td>
      <td>2.0</td>
      <td>12000</td>
      <td>850.0</td>
      <td>4 out of 4</td>
      <td>Super Area</td>
      <td>Anandbagh, Secunderabad, Moula Ali Road</td>
      <td>Hyderabad</td>
      <td>Furnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4731</th>
      <td>2022-06-24</td>
      <td>2.0</td>
      <td>13000</td>
      <td>850.0</td>
      <td>2 out of 2</td>
      <td>Super Area</td>
      <td>Manikonda, Outer Ring Road</td>
      <td>Hyderabad</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4732</th>
      <td>2022-07-08</td>
      <td>2.0</td>
      <td>7000</td>
      <td>850.0</td>
      <td>Ground out of 2</td>
      <td>Super Area</td>
      <td>Vinayaka Nagar</td>
      <td>Hyderabad</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
  </tbody>
</table>
</div>




```python
# BHK만 median을 주고 싶다면
rent_df['BHK'].fillna(rent_df['BHK'].median()).loc[na_index]
```




    425     2.0
    430     2.0
    4703    2.0
    4731    2.0
    4732    2.0
    Name: BHK, dtype: float64




```python
rent_df['Size'].fillna(rent_df['Size'].mean()).loc[na_index]
```




    425     967.477536
    430     967.477536
    4703    967.477536
    4731    967.477536
    4732    967.477536
    Name: Size, dtype: float64




```python
rent_df = rent_df.fillna(rent_df.median())
```

    C:\Users\MINUK\AppData\Local\Temp\ipykernel_11324\990425930.py:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      rent_df = rent_df.fillna(rent_df.median())
    


```python
rent_df.isna().mean()
```




    Posted On            0.0
    BHK                  0.0
    Rent                 0.0
    Size                 0.0
    Floor                0.0
    Area Type            0.0
    Area Locality        0.0
    City                 0.0
    Furnishing Status    0.0
    Tenant Preferred     0.0
    Bathroom             0.0
    Point of Contact     0.0
    dtype: float64




```python
rent_df.head()
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
      <th>Posted On</th>
      <th>BHK</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Floor</th>
      <th>Area Type</th>
      <th>Area Locality</th>
      <th>City</th>
      <th>Furnishing Status</th>
      <th>Tenant Preferred</th>
      <th>Bathroom</th>
      <th>Point of Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-05-18</td>
      <td>2.0</td>
      <td>10000</td>
      <td>1100.0</td>
      <td>Ground out of 2</td>
      <td>Super Area</td>
      <td>Bandel</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-13</td>
      <td>2.0</td>
      <td>20000</td>
      <td>800.0</td>
      <td>1 out of 3</td>
      <td>Super Area</td>
      <td>Phool Bagan, Kankurgachi</td>
      <td>Kolkata</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-05-16</td>
      <td>2.0</td>
      <td>17000</td>
      <td>1000.0</td>
      <td>1 out of 3</td>
      <td>Super Area</td>
      <td>Salt Lake City Sector 2</td>
      <td>Kolkata</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-07-04</td>
      <td>2.0</td>
      <td>10000</td>
      <td>800.0</td>
      <td>1 out of 2</td>
      <td>Super Area</td>
      <td>Dumdum Park</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-05-09</td>
      <td>2.0</td>
      <td>7500</td>
      <td>850.0</td>
      <td>1 out of 2</td>
      <td>Carpet Area</td>
      <td>South Dum Dum</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Area Type은 텍스트 형태이기 때문에 모델에서 계산이 안됨
rent_df['Area Type'].unique()
```




    array(['Super Area', 'Carpet Area', 'Built Area'], dtype=object)




```python
rent_df['Area Type'].nunique() # 3가지
```




    3




```python
rent_df['Area Type'].value_counts()
```




    Super Area     2446
    Carpet Area    2298
    Built Area        2
    Name: Area Type, dtype: int64




```python
for i in ['Area Type', 'Area Locality','City','Furnishing Status','Tenant Preferred','Point of Contact']:
    print(i,rent_df[i].nunique()) # Area Locality 갯수가 너무 많아서 빼야됨
```

    Area Type 3
    Area Locality 2235
    City 6
    Furnishing Status 3
    Tenant Preferred 3
    Point of Contact 3
    


```python
rent_df.drop(['Posted On','Floor','Area Locality'],axis=1,inplace=True)
```


```python
rent_df.head()
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
      <th>BHK</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Area Type</th>
      <th>City</th>
      <th>Furnishing Status</th>
      <th>Tenant Preferred</th>
      <th>Bathroom</th>
      <th>Point of Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>10000</td>
      <td>1100.0</td>
      <td>Super Area</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>2</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>20000</td>
      <td>800.0</td>
      <td>Super Area</td>
      <td>Kolkata</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>17000</td>
      <td>1000.0</td>
      <td>Super Area</td>
      <td>Kolkata</td>
      <td>Semi-Furnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>10000</td>
      <td>800.0</td>
      <td>Super Area</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors/Family</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>7500</td>
      <td>850.0</td>
      <td>Carpet Area</td>
      <td>Kolkata</td>
      <td>Unfurnished</td>
      <td>Bachelors</td>
      <td>1</td>
      <td>Contact Owner</td>
    </tr>
  </tbody>
</table>
</div>




```python
#컬럼이 많은 경우 첫번째 열을 삭제해주는 것이 속도상으로 유리
pd.get_dummies(rent_df,columns=['Area Type','City','Furnishing Status','Tenant Preferred','Point of Contact'],drop_first=1)
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
      <th>BHK</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Bathroom</th>
      <th>Area Type_Carpet Area</th>
      <th>Area Type_Super Area</th>
      <th>City_Chennai</th>
      <th>City_Delhi</th>
      <th>City_Hyderabad</th>
      <th>City_Kolkata</th>
      <th>City_Mumbai</th>
      <th>Furnishing Status_Semi-Furnished</th>
      <th>Furnishing Status_Unfurnished</th>
      <th>Tenant Preferred_Bachelors/Family</th>
      <th>Tenant Preferred_Family</th>
      <th>Point of Contact_Contact Builder</th>
      <th>Point of Contact_Contact Owner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>10000</td>
      <td>1100.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>20000</td>
      <td>800.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>17000</td>
      <td>1000.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>10000</td>
      <td>800.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>7500</td>
      <td>850.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>4741</th>
      <td>2.0</td>
      <td>15000</td>
      <td>1000.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4742</th>
      <td>3.0</td>
      <td>29000</td>
      <td>2000.0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4743</th>
      <td>3.0</td>
      <td>35000</td>
      <td>1750.0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4744</th>
      <td>3.0</td>
      <td>45000</td>
      <td>1500.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4745</th>
      <td>2.0</td>
      <td>15000</td>
      <td>1000.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>4746 rows × 17 columns</p>
</div>




```python
rent_df=pd.get_dummies(rent_df,columns=['Area Type','City','Furnishing Status','Tenant Preferred','Point of Contact'])
```


```python
rent_df.head()
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
      <th>BHK</th>
      <th>Rent</th>
      <th>Size</th>
      <th>Bathroom</th>
      <th>Area Type_Built Area</th>
      <th>Area Type_Carpet Area</th>
      <th>Area Type_Super Area</th>
      <th>City_Bangalore</th>
      <th>City_Chennai</th>
      <th>City_Delhi</th>
      <th>...</th>
      <th>City_Mumbai</th>
      <th>Furnishing Status_Furnished</th>
      <th>Furnishing Status_Semi-Furnished</th>
      <th>Furnishing Status_Unfurnished</th>
      <th>Tenant Preferred_Bachelors</th>
      <th>Tenant Preferred_Bachelors/Family</th>
      <th>Tenant Preferred_Family</th>
      <th>Point of Contact_Contact Agent</th>
      <th>Point of Contact_Contact Builder</th>
      <th>Point of Contact_Contact Owner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>10000</td>
      <td>1100.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>2.0</td>
      <td>20000</td>
      <td>800.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>17000</td>
      <td>1000.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>10000</td>
      <td>800.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <th>4</th>
      <td>2.0</td>
      <td>7500</td>
      <td>850.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
X = rent_df.drop('Rent',axis=1)
y= rent_df['Rent']
```


```python
X
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
      <th>BHK</th>
      <th>Size</th>
      <th>Bathroom</th>
      <th>Area Type_Built Area</th>
      <th>Area Type_Carpet Area</th>
      <th>Area Type_Super Area</th>
      <th>City_Bangalore</th>
      <th>City_Chennai</th>
      <th>City_Delhi</th>
      <th>City_Hyderabad</th>
      <th>...</th>
      <th>City_Mumbai</th>
      <th>Furnishing Status_Furnished</th>
      <th>Furnishing Status_Semi-Furnished</th>
      <th>Furnishing Status_Unfurnished</th>
      <th>Tenant Preferred_Bachelors</th>
      <th>Tenant Preferred_Bachelors/Family</th>
      <th>Tenant Preferred_Family</th>
      <th>Point of Contact_Contact Agent</th>
      <th>Point of Contact_Contact Builder</th>
      <th>Point of Contact_Contact Owner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>1100.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>2.0</td>
      <td>800.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>1000.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>800.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <th>4</th>
      <td>2.0</td>
      <td>850.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <th>4741</th>
      <td>2.0</td>
      <td>1000.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4742</th>
      <td>3.0</td>
      <td>2000.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4743</th>
      <td>3.0</td>
      <td>1750.0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4744</th>
      <td>3.0</td>
      <td>1500.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4745</th>
      <td>2.0</td>
      <td>1000.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>4746 rows × 21 columns</p>
</div>




```python
y
```




    0       10000
    1       20000
    2       17000
    3       10000
    4        7500
            ...  
    4741    15000
    4742    29000
    4743    35000
    4744    45000
    4745    15000
    Name: Rent, Length: 4746, dtype: int64




```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
```


```python
X_train.shape, X_test.shape
```




    ((3322, 21), (1424, 21))




```python
y_train.shape, y_test.shape
```




    ((3322,), (1424,))



# 1. 선형 회귀(Linear Regression)
* 독립변수(x)로 종속변수 (y)를 예측하는것


```python
#독립변수(x)로 종속변수 (y)를 예측하는 것
from sklearn.linear_model import LinearRegression
```


```python
lr = LinearRegression()
```


```python
lr.fit(X_train,y_train)
```




    LinearRegression()




```python
pred = lr.predict(X_test)
```

# 2. 평가 지표 만들기

## 2-1.MSE(Mean Squared Error)
* 예측값과 실제값의 차이에 대한 제곱에 대하여 평균을 낸 값
* ${(\frac{1}{n})\sum_{i=1}^{n}(y_{i} - x_{i})^{2}}$



```python
p=np.array([3,4,5])
act=np.array([1,2,3])
```


```python
def my_mse(pred,actual):
    return((pred-actual)**2).mean()
```


```python
my_mse(p,act)
```




    4.0



## 2-2. MAE(Mean Absolute Error)
* 예측값과 실제값의 차이에 대한 제곱에 대하여 평균을 낸 값
* $(\frac{1}{n})\sum_{i=1}^{n}\left | y_{i} - x_{i} \right |$


```python
def my_mae(pred,actual):
    return np.abs(pred-actual).mean()
```


```python
my_mae(p,act)
```




    2.0



## 2-3. RMSE(Root Mean Squared Error)
* 예측값과 실제값의 차이에 대한 제곱에 대하여 평균을 낸 후 루트를 씌운 값
* $\sqrt{(\frac{1}{n})\sum_{i=1}^{n}(y_{i} - x_{i})^{2}}$



```python
def my_rmse(pred,actual):
    return np.sqrt(my_mse(pred,actual))
```


```python
my_rmse(p,act)
```




    2.0




```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
```


```python
mean_squared_error(p,act)
```




    4.0




```python
mean_absolute_error(p,act)
```




    2.0




```python
mean_squared_error(p,act,squared=False) #rmse
```




    2.0



# 3. 평가지표 적용


```python
mean_squared_error(y_test,pred)
```




    1509545939.0374868




```python
mean_absolute_error(y_test,pred)
```




    22758.09248383841




```python
mean_squared_error(y_test,pred,squared=False)
```




    38852.8755568682




```python
X_train.loc[1837]
```




    BHK                                     3.0
    Size                                 2500.0
    Bathroom                                3.0
    Area Type_Built Area                    0.0
    Area Type_Carpet Area                   1.0
    Area Type_Super Area                    0.0
    City_Bangalore                          1.0
    City_Chennai                            0.0
    City_Delhi                              0.0
    City_Hyderabad                          0.0
    City_Kolkata                            0.0
    City_Mumbai                             0.0
    Furnishing Status_Furnished             0.0
    Furnishing Status_Semi-Furnished        1.0
    Furnishing Status_Unfurnished           0.0
    Tenant Preferred_Bachelors              1.0
    Tenant Preferred_Bachelors/Family       0.0
    Tenant Preferred_Family                 0.0
    Point of Contact_Contact Agent          1.0
    Point of Contact_Contact Builder        0.0
    Point of Contact_Contact Owner          0.0
    Name: 1837, dtype: float64




```python
#아웃라이어 제거
X_train.drop(1837,inplace=True)
y_train.drop(1837,inplace=True)
```


```python
lr.fit(X_train,y_train)
```




    LinearRegression()




```python
new_pred = lr.predict(X_test)
```


```python
mean_squared_error(y_test,new_pred,squared=True)
```




    1465329699.5746772



# 4. log 활용하기


```python
a = [1,2,3,4,5]
b=[1,10,100,1000,10000]
```


```python
sns.lineplot(x=a,y=b)
```




    <AxesSubplot:>




![png](output_83_1.png)



```python
b_log = np.log(b) # 선형에 가깝게 곡선을 외곡시킴
```


```python
sns.lineplot(x=a,y=b_log)
```




    <AxesSubplot:>




![png](output_85_1.png)



```python
b_log
```




    array([0.        , 2.30258509, 4.60517019, 6.90775528, 9.21034037])




```python
np.exp(b_log)
```




    array([1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04])




```python
y_train_log = np.log(y_train)
```


```python
lr.fit(X_train,y_train_log)
```




    LinearRegression()




```python
newnew_pred = lr.predict(X_test)
```


```python
pred_exp = np.exp(newnew_pred)
```


```python
mean_squared_error(y_test,pred_exp,squared=True)
```




    1025266439.2592406



> 데이터의 분포가 선형일 때는 로그를 사용하는 것이 좋지 않지만, 비선형일 때는 로그를 사용하는 것이 성능에 좋음


```python

```
