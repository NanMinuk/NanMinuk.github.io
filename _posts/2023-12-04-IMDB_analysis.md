---
title: Sentiment Analysis by LSTM
tags: NLP
typora-root-url: ../
---

# 영화평 분류 - Sentiment Analysis

- IMDB (Internet Movie Database, https://www.imdb.com/)  Dataset


- 각 25,000 개의 training/testing set 으로 구성된 IMDB 영화관람평  


- “imdb_reviews” – encoding 되어있지 않은 string 형태의 data  


- label : positive, negative binary classification

## imdb_reviews (string data) 이용 


```python
 import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
```


```python
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']
```


```python
len(train_dataset), len(test_dataset)
```




    (25000, 25000)



### tfds 는 tf.Tensor 반환


```python
for input, label in dataset['test']:
    print(label)
    print(input)
    break
```

    tf.Tensor(1, shape=(), dtype=int64)
    tf.Tensor(b"There are films that make careers. For George Romero, it was NIGHT OF THE LIVING DEAD; for Kevin Smith, CLERKS; for Robert Rodriguez, EL MARIACHI. Add to that list Onur Tukel's absolutely amazing DING-A-LING-LESS. Flawless film-making, and as assured and as professional as any of the aforementioned movies. I haven't laughed this hard since I saw THE FULL MONTY. (And, even then, I don't think I laughed quite this hard... So to speak.) Tukel's talent is considerable: DING-A-LING-LESS is so chock full of double entendres that one would have to sit down with a copy of this script and do a line-by-line examination of it to fully appreciate the, uh, breadth and width of it. Every shot is beautifully composed (a clear sign of a sure-handed director), and the performances all around are solid (there's none of the over-the-top scenery chewing one might've expected from a film like this). DING-A-LING-LESS is a film whose time has come.", shape=(), dtype=string)


## keras.preprocessing module 을 이용하여 tokenize 및 padding
- keras.preprocessing.text.Tokenizer  

- keras.preprocessing.sequence.pad_sequence


```python
train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

for sent, label in train_dataset:
    train_sentences.append(str(sent.numpy()))
    train_labels.append(label.numpy())

for sent, label in test_dataset:
    test_sentences.append(str(sent.numpy()))
    test_labels.append(label.numpy())

print(train_labels[-1])
print(train_sentences[-1])

print(test_labels[-1])
print(test_sentences[-1])
```

    1
    b'First of all, Riget is wonderful. Good comedy and mystery thriller at the same time. Nice combination of strange \'dogma\' style of telling the story together with good music and great actors. But unfortunately there\'s no \'the end\'. As for me it\'s unacceptable. I was thinking... how it will be possible to continue the story without Helmer and Drusse? ...and I have some idea. I think Lars should make RIGET III a little bit different. I\'m sure that 3rd part without Helmer wouldn\'t be the same. So here\'s my suggestion. Mayble little bit stupid, maybe not. I know that Lars likes to experiment. So why not to make small experiment with Riget3? I think the only solution here is to create puppet-driven animation (like for example "team America" by Trey Parker) or even computer 3d animation. I know it\'s not the same as real actors, but in principle I believe it could work... only this way it\'s possible to make actors alive again. For Riget fans this shouldn\'t be so big difference - if the animation will be done in good way average \'watcher\' will consider it normal just after first few shots of the movie. The most important thing now is the story. It\'s completely understandable that it\'s not possible to create Riget 3 with the actors nowadays. So why not to play with animation? And... look for the possibilities that it gives to you! Even marketing one! Great director finishes his trilogy after 10 years using puppet animation. Just dreams?<br /><br />I hope to see Riget 3 someday... or even to see just the script. I\'m curious how the story ends... and as I expect- everybody here do.<br /><br />greets, slaj<br /><br />ps: I\'m not talking about the "kingdom hospital" by Stephen King ;-)'
    1
    b"They just don't make cartoons like they used to. This one had wit, great characters, and the greatest ensemble of voice over artists ever assembled for a daytime cartoon show. This still remains as one of the highest rated daytime cartoon shows, and one of the most honored, winning several Emmy Awards."


## train_labels, test_labels  list를 numpy array 로 변환


```python
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print(train_labels.shape)
print(test_labels.shape)
```

    (25000,)
    (25000,)


## train_sentences, test_sentences text 를 sequence 로 변환 


```python
vocab_size = 10000

tokenizer = Tokenizer(num_words = vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_sentences)
```


```python
train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

print(train_sequences[0])
print(test_sequences[0])
```

    [59, 12, 14, 35, 439, 400, 18, 174, 29, 1, 9, 33, 1378, 3401, 42, 496, 1, 197, 25, 88, 156, 19, 12, 211, 340, 29, 70, 248, 213, 9, 486, 62, 70, 88, 116, 99, 24, 5740, 12, 3317, 657, 777, 12, 18, 7, 35, 406, 8228, 178, 2477, 426, 2, 92, 1253, 140, 72, 149, 55, 2, 1, 7525, 72, 229, 70, 2962, 16, 1, 2880, 1, 1, 1506, 4998, 3, 40, 3947, 119, 1608, 17, 3401, 14, 163, 19, 4, 1253, 927, 7986, 9, 4, 18, 13, 14, 4200, 5, 102, 148, 1237, 11, 240, 692, 13, 44, 25, 101, 39, 12, 7232, 1, 39, 1378, 1, 52, 409, 11, 99, 1214, 874, 145, 10]
    [59, 44, 25, 109, 13, 97, 4115, 16, 742, 4370, 10, 14, 316, 5, 2, 593, 354, 16, 1864, 1212, 1, 16, 680, 7499, 5595, 1, 773, 6, 13, 1037, 1, 1, 439, 491, 1, 4, 1, 334, 3610, 20, 229, 3, 15, 5796, 3, 15, 1646, 15, 102, 5, 2, 3597, 101, 11, 1450, 1528, 12, 251, 235, 11, 216, 2, 377, 6429, 3, 62, 95, 11, 174, 105, 11, 1528, 180, 12, 251, 37, 6, 1144, 1, 682, 7, 4452, 1, 4, 1, 334, 7, 37, 8367, 377, 5, 1420, 1, 13, 30, 64, 28, 6, 874, 181, 17, 4, 1050, 5, 12, 224, 3, 83, 4, 353, 33, 353, 5229, 5, 10, 6, 1340, 1160, 2, 5738, 1, 3, 1, 5, 10, 175, 328, 7, 1319, 3989, 4, 798, 1946, 5, 4, 250, 2710, 158, 3, 2, 361, 31, 187, 25, 1170, 499, 610, 5, 2, 122, 2, 356, 1398, 7725, 30, 1, 881, 38, 4, 20, 39, 12, 1, 4, 1, 334, 7, 4, 20, 634, 60, 48, 214]



```python
plt.hist([len(s) for s in train_sequences] + [len(s) for s in test_sequences], bins=50);
```

![output_14_0](/images/2023-12-05-IMDB_analysis/output_14_0.png)

```python
max_length = 150

train_padded = pad_sequences(train_sequences,maxlen=max_length, truncating='post', padding='post')

test_padded = pad_sequences(test_sequences,maxlen=max_length, truncating='post', padding='post')

print(train_padded.shape)
print(test_padded.shape)
print(train_padded[0])
print(test_padded[0])
```

    (25000, 150)
    (25000, 150)
    [  59   12   14   35  439  400   18  174   29    1    9   33 1378 3401
       42  496    1  197   25   88  156   19   12  211  340   29   70  248
      213    9  486   62   70   88  116   99   24 5740   12 3317  657  777
       12   18    7   35  406 8228  178 2477  426    2   92 1253  140   72
      149   55    2    1 7525   72  229   70 2962   16    1 2880    1    1
     1506 4998    3   40 3947  119 1608   17 3401   14  163   19    4 1253
      927 7986    9    4   18   13   14 4200    5  102  148 1237   11  240
      692   13   44   25  101   39   12 7232    1   39 1378    1   52  409
       11   99 1214  874  145   10    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0]
    [  59   44   25  109   13   97 4115   16  742 4370   10   14  316    5
        2  593  354   16 1864 1212    1   16  680 7499 5595    1  773    6
       13 1037    1    1  439  491    1    4    1  334 3610   20  229    3
       15 5796    3   15 1646   15  102    5    2 3597  101   11 1450 1528
       12  251  235   11  216    2  377 6429    3   62   95   11  174  105
       11 1528  180   12  251   37    6 1144    1  682    7 4452    1    4
        1  334    7   37 8367  377    5 1420    1   13   30   64   28    6
      874  181   17    4 1050    5   12  224    3   83    4  353   33  353
     5229    5   10    6 1340 1160    2 5738    1    3    1    5   10  175
      328    7 1319 3989    4  798 1946    5    4  250 2710  158    3    2
      361   31  187   25 1170  499  610    5    2  122]


- sequence data 를 sentence 로 reverse conversion


```python
reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])

def decode_review(sequence):
    return ' '.join([tokenizer.index_word.get(i, '<pad>') for i in sequence])

print(decode_review(train_padded[0]))
print()
print(train_sentences[0])
```

    b this was an absolutely terrible movie don't be <OOV> in by christopher walken or michael <OOV> both are great actors but this must simply be their worst role in history even their great acting could not redeem this movie's ridiculous storyline this movie is an early nineties us propaganda piece the most pathetic scenes were those when the <OOV> rebels were making their cases for <OOV> maria <OOV> <OOV> appeared phony and her pseudo love affair with walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning i am disappointed that there are movies like this ruining <OOV> like christopher <OOV> good name i could barely sit through it <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
    
    b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it."


## model define


```python
model = Sequential([
    Embedding(vocab_size+1, 64),
    Bidirectional(tf.keras.layers.LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, None, 64)          640064    
                                                                     
     bidirectional (Bidirectiona  (None, 128)              66048     
     l)                                                              
                                                                     
     dense (Dense)               (None, 64)                8256      
                                                                     
     dense_1 (Dense)             (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 714,433
    Trainable params: 714,433
    Non-trainable params: 0
    _________________________________________________________________



```python
%%time
num_epochs = 30
history = model.fit(train_padded, train_labels, epochs=num_epochs, batch_size=128,
                validation_data=(test_padded, test_labels), verbose=1)
```

    Epoch 1/30
    196/196 [==============================] - 11s 34ms/step - loss: 0.4854 - accuracy: 0.7524 - val_loss: 0.3791 - val_accuracy: 0.8358
    Epoch 2/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.2780 - accuracy: 0.8899 - val_loss: 0.3847 - val_accuracy: 0.8384
    Epoch 3/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.2085 - accuracy: 0.9224 - val_loss: 0.3983 - val_accuracy: 0.8287
    Epoch 4/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.1630 - accuracy: 0.9403 - val_loss: 0.4758 - val_accuracy: 0.8135
    Epoch 5/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.1347 - accuracy: 0.9518 - val_loss: 0.4993 - val_accuracy: 0.8180
    Epoch 6/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.1032 - accuracy: 0.9640 - val_loss: 0.5570 - val_accuracy: 0.8024
    Epoch 7/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0779 - accuracy: 0.9734 - val_loss: 0.7996 - val_accuracy: 0.8136
    Epoch 8/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0421 - accuracy: 0.9864 - val_loss: 0.8203 - val_accuracy: 0.8082
    Epoch 9/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0328 - accuracy: 0.9900 - val_loss: 1.0368 - val_accuracy: 0.8034
    Epoch 10/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0347 - accuracy: 0.9887 - val_loss: 0.9215 - val_accuracy: 0.8112
    Epoch 11/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0219 - accuracy: 0.9932 - val_loss: 1.1193 - val_accuracy: 0.8117
    Epoch 12/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0262 - accuracy: 0.9910 - val_loss: 1.2774 - val_accuracy: 0.7979
    Epoch 13/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0195 - accuracy: 0.9940 - val_loss: 1.1893 - val_accuracy: 0.7851
    Epoch 14/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0179 - accuracy: 0.9939 - val_loss: 1.2443 - val_accuracy: 0.8045
    Epoch 15/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0136 - accuracy: 0.9959 - val_loss: 1.2135 - val_accuracy: 0.8038
    Epoch 16/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0076 - accuracy: 0.9976 - val_loss: 1.2430 - val_accuracy: 0.8002
    Epoch 17/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0131 - accuracy: 0.9960 - val_loss: 1.2519 - val_accuracy: 0.8002
    Epoch 18/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0107 - accuracy: 0.9963 - val_loss: 1.3192 - val_accuracy: 0.7987
    Epoch 19/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0161 - accuracy: 0.9943 - val_loss: 1.3434 - val_accuracy: 0.7966
    Epoch 20/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0119 - accuracy: 0.9963 - val_loss: 1.5562 - val_accuracy: 0.8055
    Epoch 21/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0028 - accuracy: 0.9992 - val_loss: 1.6675 - val_accuracy: 0.8079
    Epoch 22/30
    196/196 [==============================] - 6s 29ms/step - loss: 9.9940e-04 - accuracy: 0.9998 - val_loss: 1.5337 - val_accuracy: 0.7897
    Epoch 23/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0048 - accuracy: 0.9984 - val_loss: 1.6966 - val_accuracy: 0.8113
    Epoch 24/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0171 - accuracy: 0.9946 - val_loss: 1.3005 - val_accuracy: 0.8033
    Epoch 25/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0185 - accuracy: 0.9938 - val_loss: 1.2342 - val_accuracy: 0.7999
    Epoch 26/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0135 - accuracy: 0.9958 - val_loss: 1.1881 - val_accuracy: 0.7955
    Epoch 27/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0069 - accuracy: 0.9976 - val_loss: 1.3981 - val_accuracy: 0.8041
    Epoch 28/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0084 - accuracy: 0.9973 - val_loss: 1.3295 - val_accuracy: 0.7887
    Epoch 29/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0083 - accuracy: 0.9972 - val_loss: 1.4624 - val_accuracy: 0.7977
    Epoch 30/30
    196/196 [==============================] - 6s 29ms/step - loss: 0.0040 - accuracy: 0.9987 - val_loss: 1.5223 - val_accuracy: 0.7984
    CPU times: user 3min 3s, sys: 25.5 s, total: 3min 28s
    Wall time: 2min 56s



```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_xlabel('Epochs')
ax1.set_ylabel('accuracy')
ax1.legend(['accuarcy', 'val_accuracy'])

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_xlabel('Epochs')
ax2.set_ylabel('loss')
ax2.legend(['loss', 'val_loss'])
plt.show()
```


![output_21_0](/images/2023-12-05-IMDB_analysis/output_21_0.png)



```python
sample_text = ['The movei was terrible. I would not recommend the movie']
# sample_text = ['The movei was fantastic. I would recommend the movie']
# sample_text = ['The animation and graphics were out of this world']
sample_seq = tokenizer.texts_to_sequences(sample_text)
sample_padded = pad_sequences(sample_seq, maxlen=max_length, padding='post', truncating='post')
sample_padded
```




    array([[  2,   1,  14, 400,  11,  64,  24, 392,   2,  18,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0]], dtype=int32)




```python
model.predict([sample_padded])
```




    array([[0.01155183]], dtype=float32)




```python

```
