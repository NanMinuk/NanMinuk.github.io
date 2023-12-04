---
title: Tenserflow
tags: Machine_Learnig
typora-root-url: ../
---

# 1. 텐서플로우(Tenserflow)
* 텐서플로우는 ML/DL 모델을 개발하고 학습시키는데 도움이 되는 핵심 오픈 소스 라이브러리
* 텐서플로우 2.x에서는 케라스를 딥러닝 공식 API로 채택하였고, 텐서플로우 내의 하나의 프레임워크로 개발하고 있음


```python
import tensorflow as tf
```


```python
print(tf.__version__)
```

    2.11.0
    

### 1-1. Tensor
* Tensor는 multi-dimensional array를 나타내는 말
* Tensorflow의 기본 datatype


```python
a = tf.constant([10,3],dtype=tf.float32)
print(a)

b=tf.constant('Hello TensorFlow!')
print(b)
```

    tf.Tensor([10.  3.], shape=(2,), dtype=float32)
    tf.Tensor(b'Hello TensorFlow!', shape=(), dtype=string)
    


```python
c= tf.constant([[1.0,2.0],[3.0,4.0]])
print(c)
print(type(c))
```

    tf.Tensor(
    [[1. 2.]
     [3. 4.]], shape=(2, 2), dtype=float32)
    <class 'tensorflow.python.framework.ops.EagerTensor'>
    


```python
import numpy as np
```


```python
x_np = np.array([[1.0,2.0],[3.0,4.0]])

x_list = [[1.0,2.0],[3.0,4.0]]

print(type(x_np))
print(type(x_list))
```

    <class 'numpy.ndarray'>
    <class 'list'>
    


```python
#ndarray를 tensor로 변환
x_np_tf = tf.convert_to_tensor(x_np)
print(type(x_np_tf))

# list를 tensor로 변환
x_list_tf = tf.convert_to_tensor(x_list)
print(type(x_list_tf))
```

    <class 'tensorflow.python.framework.ops.EagerTensor'>
    <class 'tensorflow.python.framework.ops.EagerTensor'>
    


```python
# tensor를 numpy의 ndarray로 변환
print(x_np_tf.numpy())
print(type(x_np_tf.numpy()))
```

    [[1. 2.]
     [3. 4.]]
    <class 'numpy.ndarray'>
    

## 1-2. 텐서플로우 함수


```python
a= tf.ones((2,3))
print(a)

b=tf.zeros((2,3))
print(b)

c=tf.fill((2,3),10)
print(c)
# 매개변수로 전달된 텐서 행렬을 복사하여 같은 shape의 행렬을 생성(데이터는 복사하지 않음)
d= tf.zeros_like(c)
print(d)

e = tf.ones_like(c)
print(e)
```

    tf.Tensor(
    [[1. 1. 1.]
     [1. 1. 1.]], shape=(2, 3), dtype=float32)
    tf.Tensor(
    [[0. 0. 0.]
     [0. 0. 0.]], shape=(2, 3), dtype=float32)
    tf.Tensor(
    [[10 10 10]
     [10 10 10]], shape=(2, 3), dtype=int32)
    tf.Tensor(
    [[0 0 0]
     [0 0 0]], shape=(2, 3), dtype=int32)
    tf.Tensor(
    [[1 1 1]
     [1 1 1]], shape=(2, 3), dtype=int32)
    


```python
f= tf.range(10) # 범위를 생성
print(f)
```

    tf.Tensor([0 1 2 3 4 5 6 7 8 9], shape=(10,), dtype=int32)
    


```python
# 0~1 사이의 랜덤한 값을 2행 2열로 추출
g= tf.random.uniform((2,2))
print(g)

# 정규분포 난수를 2행 2열로 추출
h=tf.random.normal((2,2))
print(h)
```

    tf.Tensor(
    [[0.27105212 0.9374579 ]
     [0.11074495 0.67587566]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[-0.1294738   0.15022035]
     [-0.9316058   0.5288355 ]], shape=(2, 2), dtype=float32)
    

### 1-3. Tensor  속성


```python
tensor = tf.random.normal((3,4))
print(f'shape:({tensor.shape}')
print(f'Datatype:({tensor.dtype}')
```

    shape:((3, 4)
    Datatype:(<dtype: 'float32'>
    


```python
tensor = tf.reshape(tensor, (4,3))
print(f'shape:({tensor.shape}')
tensor = tf.cast(tensor , tf.int32)
print(f'Datatype:({tensor.dtype}')
print(tensor)
```

    shape:((4, 3)
    Datatype:(<dtype: 'int32'>
    tf.Tensor(
    [[ 0  1  0]
     [ 0  0  0]
     [ 0  1 -1]
     [ 0 -1  1]], shape=(4, 3), dtype=int32)
    

### 1-4 . Variable
* Variable은 변할 수 있는 상태를 저장하는데 사용되는 특별한 텐서
* 딥러닝에서는 학습해야하는 가중치(weight, bias)들을 variable 생성


```python
tensor = tf.ones((3,4))
print(tensor)

tensor[0,0]=100 #TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment

```

    tf.Tensor(
    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]], shape=(3, 4), dtype=float32)
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_18420\3154442876.py in <module>
          2 print(tensor)
          3 
    ----> 4 tensor[0,0]=100 #TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
    

    TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment



```python
var = tf.Variable(tensor)
print(var)

#assign(): variable에 값을 설정
var[0,0].assign(100)
print(var)
```

    <tf.Variable 'Variable:0' shape=(3, 4) dtype=float32, numpy=
    array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]], dtype=float32)>
    <tf.Variable 'Variable:0' shape=(3, 4) dtype=float32, numpy=
    array([[100.,   1.,   1.,   1.],
           [  1.,   1.,   1.,   1.],
           [  1.,   1.,   1.,   1.]], dtype=float32)>
    


```python
value1 = tf.random.normal(shape=(2,2))
value1 = tf.Variable(value1)
print(value1)

value2 = tf.random.normal(shape=(2,2))
print(value2)
value1.assign(value2)
print(value1)

# assign_add():variable에 값을 더해주는 함수
# assign_sub():variable에 값을 빼주는 함수
value3 = tf.ones(shape=(2,2))
value1.assign_add(value3)
print(value1)
```

    <tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
    array([[ 0.09566392, -0.3559116 ],
           [ 0.25090444,  0.0406981 ]], dtype=float32)>
    tf.Tensor(
    [[ 0.41741538 -0.81282085]
     [-0.3801746  -0.453481  ]], shape=(2, 2), dtype=float32)
    <tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
    array([[ 0.41741538, -0.81282085],
           [-0.3801746 , -0.453481  ]], dtype=float32)>
    <tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
    array([[1.4174154 , 0.18717915],
           [0.61982536, 0.54651904]], dtype=float32)>
    

### 1-5 . indexing, slicing


```python
a = tf.range(1,13)
print(a)

a= tf.reshape(a, (3,4))
print(a)
```

    tf.Tensor([ 1  2  3  4  5  6  7  8  9 10 11 12], shape=(12,), dtype=int32)
    tf.Tensor(
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]], shape=(3, 4), dtype=int32)
    


```python
# 인덱싱을 하면 차원이 감소하게 됨
print(a[1])
print(a[0,-1])
```

    tf.Tensor([5 6 7 8], shape=(4,), dtype=int32)
    tf.Tensor(4, shape=(), dtype=int32)
    


```python
# 슬라이싱은 차원이 유지됨
print(a[1:-1])
print(a[:2,2:])
```

    tf.Tensor([[5 6 7 8]], shape=(1, 4), dtype=int32)
    tf.Tensor(
    [[3 4]
     [7 8]], shape=(2, 2), dtype=int32)
    

### 1-6. 차원 바꾸기


```python
a= tf.range(16)
print(a)
a= tf.reshape(a,(2,2,-1)) # -1: 자동으로 설정하라는 뜻(넣기 귀찮거나 모를때/ 여기서는 4)
print(a)
```

    tf.Tensor([ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15], shape=(16,), dtype=int32)
    tf.Tensor(
    [[[ 0  1  2  3]
      [ 4  5  6  7]]
    
     [[ 8  9 10 11]
      [12 13 14 15]]], shape=(2, 2, 4), dtype=int32)
    


```python
# transpose(): 행렬의 차원을 인덱스로 변환
b= tf.transpose(a, (2,0,1))
print(b)
```

    tf.Tensor(
    [[[ 0  4]
      [ 8 12]]
    
     [[ 1  5]
      [ 9 13]]
    
     [[ 2  6]
      [10 14]]
    
     [[ 3  7]
      [11 15]]], shape=(4, 2, 2), dtype=int32)
    

### 1-7. Tensor 연산


```python
x= tf.constant([[1,2],[3,4]],dtype = tf.float32)
y= tf.constant([[1,2],[3,4]],dtype = tf.float32)
print(x)
print(y)
```

    tf.Tensor(
    [[1. 2.]
     [3. 4.]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[1. 2.]
     [3. 4.]], shape=(2, 2), dtype=float32)
    


```python
print(tf.add(x,y))
print(tf.subtract(x,y))
print(tf.multiply(x,y))
print(tf.divide(x,y))
print(tf.matmul(x,y)) # 내적
```

    tf.Tensor(
    [[2. 4.]
     [6. 8.]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[0. 0.]
     [0. 0.]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[ 1.  4.]
     [ 9. 16.]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[1. 1.]
     [1. 1.]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[ 7. 10.]
     [15. 22.]], shape=(2, 2), dtype=float32)
    


```python
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x @ y) # 내적
```

    tf.Tensor(
    [[2. 4.]
     [6. 8.]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[0. 0.]
     [0. 0.]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[ 1.  4.]
     [ 9. 16.]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[1. 1.]
     [1. 1.]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[ 7. 10.]
     [15. 22.]], shape=(2, 2), dtype=float32)
    


```python
z = tf.range(1,11)
print(z)
z=tf.reshape(z,(2,5))
print(z)
```

    tf.Tensor([ 1  2  3  4  5  6  7  8  9 10], shape=(10,), dtype=int32)
    tf.Tensor(
    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]], shape=(2, 5), dtype=int32)
    


```python
#reduce_sum(): 요소의 총 합계를 구함
print(tf.reduce_sum(z))

print(tf.reduce_sum(z,axis=0)) # 행

print(tf.reduce_sum(z,axis=1)) # 열

print(tf.reduce_sum(z,axis = -1)) # 마지막 방향(여기서는 열)
```

    tf.Tensor(55, shape=(), dtype=int32)
    tf.Tensor([ 7  9 11 13 15], shape=(5,), dtype=int32)
    tf.Tensor([15 40], shape=(2,), dtype=int32)
    tf.Tensor([15 40], shape=(2,), dtype=int32)
    


```python
# concat(): 행과 열을 합침
cc= tf.concat([z,z], axis=0) # 행
print(cc)

cc= tf.concat([z,z], axis=1) # 열
print(cc)

```

    tf.Tensor(
    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]
     [ 1  2  3  4  5]
     [ 6  7  8  9 10]], shape=(4, 5), dtype=int32)
    tf.Tensor(
    [[ 1  2  3  4  5  1  2  3  4  5]
     [ 6  7  8  9 10  6  7  8  9 10]], shape=(2, 10), dtype=int32)
    


```python

```
