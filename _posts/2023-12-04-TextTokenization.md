---
title: 텍스트 전처리
tags: NLP
typora-root-url: ../
---

## 텍스트 사전 준비 작업(텍스트 전처리) - 텍스트 정규화

### Text Tokenization

**문장 토큰화**


```python
import nltk
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\MinUk\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping tokenizers\punkt.zip.
    




    True




```python
from nltk import sent_tokenize
text_sample = 'The Matrix is everywhere its all around us, here even in this room.  \
              You can see it out your window or on your television. \
               You feel it when you go to work, or go to church or pay your taxes.'
sentences = sent_tokenize(text=text_sample)
print(type(sentences),len(sentences))
print(sentences)
```

    <class 'list'> 3
    ['The Matrix is everywhere its all around us, here even in this room.', 'You can see it out your window or on your television.', 'You feel it when you go to work, or go to church or pay your taxes.']
    

**단어 토큰화**


```python
from nltk import word_tokenize

sentence = "The Matrix is everywhere its all around us, here even in this room."
words = word_tokenize(sentence)
print(type(words), len(words))
print(words)
```

    <class 'list'> 15
    ['The', 'Matrix', 'is', 'everywhere', 'its', 'all', 'around', 'us', ',', 'here', 'even', 'in', 'this', 'room', '.']
    

**여러 문장들에 대한 단어 토큰화**


```python
from nltk import word_tokenize, sent_tokenize

#여러개의 문장으로 된 입력 데이터를 문장별로 단어 토큰화 만드는 함수 생성
def tokenize_text(text):
    
    # 문장별로 분리 토큰
    sentences = sent_tokenize(text)
    # 분리된 문장별 단어 토큰화
    word_tokens = [word_tokenize(sentence) for sentence in sentences]
    return word_tokens

#여러 문장들에 대해 문장별 단어 토큰화 수행. 
word_tokens = tokenize_text(text_sample)
print(type(word_tokens),len(word_tokens))
print(word_tokens)
```

    <class 'list'> 3
    [['The', 'Matrix', 'is', 'everywhere', 'its', 'all', 'around', 'us', ',', 'here', 'even', 'in', 'this', 'room', '.'], ['You', 'can', 'see', 'it', 'out', 'your', 'window', 'or', 'on', 'your', 'television', '.'], ['You', 'feel', 'it', 'when', 'you', 'go', 'to', 'work', ',', 'or', 'go', 'to', 'church', 'or', 'pay', 'your', 'taxes', '.']]
    

**n-gram**


```python
from nltk import ngrams

sentence = "The Matrix is everywhere its all around us, here even in this room."
words = word_tokenize(sentence)

all_ngrams = ngrams(words, 2)
ngrams = [ngram for ngram in all_ngrams]
print(ngrams)
```

    [('The', 'Matrix'), ('Matrix', 'is'), ('is', 'everywhere'), ('everywhere', 'its'), ('its', 'all'), ('all', 'around'), ('around', 'us'), ('us', ','), (',', 'here'), ('here', 'even'), ('even', 'in'), ('in', 'this'), ('this', 'room'), ('room', '.')]
    

### Stopwords 제거


```python
import nltk
nltk.download('stopwords')
```


```python
print('영어 stop words 갯수:',len(nltk.corpus.stopwords.words('english')))
print(nltk.corpus.stopwords.words('english')[:40])
```


```python
import nltk

stopwords = nltk.corpus.stopwords.words('english')
all_tokens = []
# 위 예제의 3개의 문장별로 얻은 word_tokens list 에 대해 stop word 제거 Loop
for sentence in word_tokens:
    filtered_words=[]
    # 개별 문장별로 tokenize된 sentence list에 대해 stop word 제거 Loop
    for word in sentence:
        #소문자로 모두 변환합니다. 
        word = word.lower()
        # tokenize 된 개별 word가 stop words 들의 단어에 포함되지 않으면 word_tokens에 추가
        if word not in stopwords:
            filtered_words.append(word)
    all_tokens.append(filtered_words)
    
print(all_tokens)
```

    [['matrix', 'everywhere', 'around', 'us', ',', 'even', 'room', '.'], ['see', 'window', 'television', '.'], ['feel', 'go', 'work', ',', 'go', 'church', 'pay', 'taxes', '.']]
    

### Stemming과 Lemmatization


```python
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

print(stemmer.stem('working'),stemmer.stem('works'),stemmer.stem('worked'))
print(stemmer.stem('amusing'),stemmer.stem('amuses'),stemmer.stem('amused'))
print(stemmer.stem('happier'),stemmer.stem('happiest'))
print(stemmer.stem('fancier'),stemmer.stem('fanciest'))
```

    work work work
    amus amus amus
    happy happiest
    fant fanciest
    


```python
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()
print(lemma.lemmatize('amusing','v'),lemma.lemmatize('amuses','v'),lemma.lemmatize('amused','v'))
print(lemma.lemmatize('happier','a'),lemma.lemmatize('happiest','a'))
print(lemma.lemmatize('fancier','a'),lemma.lemmatize('fanciest','a'))
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\MinUk\AppData\Roaming\nltk_data...
    

    amuse amuse amuse
    happy happy
    fancy fancy
    

### 8.3 Bag of Words – BOW

**사이킷런 CountVectorizer 테스트**


```python
text_sample_01 = 'The Matrix is everywhere its all around us, here even in this room. \
                  You can see it out your window or on your television. \
                  You feel it when you go to work, or go to church or pay your taxes.'
text_sample_02 = 'You take the blue pill and the story ends.  You wake in your bed and you believe whatever you want to believe\
                  You take the red pill and you stay in Wonderland and I show you how deep the rabbit-hole goes.'
text=[]
text.append(text_sample_01); text.append(text_sample_02)
print(text,"\n", len(text))
```

    ['The Matrix is everywhere its all around us, here even in this room.                   You can see it out your window or on your television.                   You feel it when you go to work, or go to church or pay your taxes.', 'You take the blue pill and the story ends.  You wake in your bed and you believe whatever you want to believe                  You take the red pill and you stay in Wonderland and I show you how deep the rabbit-hole goes.'] 
     2
    

**CountVectorizer객체 생성 후 fit(), transform()으로 텍스트에 대한 feature vectorization 수행**


```python
from sklearn.feature_extraction.text import CountVectorizer

# Count Vectorization으로 feature extraction 변환 수행. 
cnt_vect = CountVectorizer()
cnt_vect.fit(text)
```




    CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                    dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
                    lowercase=True, max_df=1.0, max_features=None, min_df=1,
                    ngram_range=(1, 1), preprocessor=None, stop_words=None,
                    strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                    tokenizer=None, vocabulary=None)




```python
ftr_vect = cnt_vect.transform(text)
```

**피처 벡터화 후 데이터 유형 및 여러 속성 확인**


```python
print(type(ftr_vect), ftr_vect.shape)
print(ftr_vect)
```

    <class 'scipy.sparse.csr.csr_matrix'> (2, 51)
      (0, 0)	1
      (0, 2)	1
      (0, 6)	1
      (0, 7)	1
      (0, 10)	1
      (0, 11)	1
      (0, 12)	1
      (0, 13)	2
      (0, 15)	1
      (0, 18)	1
      (0, 19)	1
      (0, 20)	2
      (0, 21)	1
      (0, 22)	1
      (0, 23)	1
      (0, 24)	3
      (0, 25)	1
      (0, 26)	1
      (0, 30)	1
      (0, 31)	1
      (0, 36)	1
      (0, 37)	1
      (0, 38)	1
      (0, 39)	1
      (0, 40)	2
      :	:
      (1, 1)	4
      (1, 3)	1
      (1, 4)	2
      (1, 5)	1
      (1, 8)	1
      (1, 9)	1
      (1, 14)	1
      (1, 16)	1
      (1, 17)	1
      (1, 18)	2
      (1, 27)	2
      (1, 28)	1
      (1, 29)	1
      (1, 32)	1
      (1, 33)	1
      (1, 34)	1
      (1, 35)	2
      (1, 38)	4
      (1, 40)	1
      (1, 42)	1
      (1, 43)	1
      (1, 44)	1
      (1, 47)	1
      (1, 49)	7
      (1, 50)	1
    


```python
print(cnt_vect.vocabulary_)
```

    {'the': 38, 'matrix': 22, 'is': 19, 'everywhere': 11, 'its': 21, 'all': 0, 'around': 2, 'us': 41, 'here': 15, 'even': 10, 'in': 18, 'this': 39, 'room': 30, 'you': 49, 'can': 6, 'see': 31, 'it': 20, 'out': 25, 'your': 50, 'window': 46, 'or': 24, 'on': 23, 'television': 37, 'feel': 12, 'when': 45, 'go': 13, 'to': 40, 'work': 48, 'church': 7, 'pay': 26, 'taxes': 36, 'take': 35, 'blue': 5, 'pill': 27, 'and': 1, 'story': 34, 'ends': 9, 'wake': 42, 'bed': 3, 'believe': 4, 'whatever': 44, 'want': 43, 'red': 29, 'stay': 33, 'wonderland': 47, 'show': 32, 'how': 17, 'deep': 8, 'rabbit': 28, 'hole': 16, 'goes': 14}
    


```python
cnt_vect = CountVectorizer(max_features=5, stop_words='english')
cnt_vect.fit(text)
ftr_vect = cnt_vect.transform(text)
print(type(ftr_vect), ftr_vect.shape)
print(cnt_vect.vocabulary_)

```

    <class 'scipy.sparse.csr.csr_matrix'> (2, 5)
    {'window': 4, 'pill': 1, 'wake': 2, 'believe': 0, 'want': 3}
    

**ngram_range 확인**


```python
cnt_vect = CountVectorizer(ngram_range=(1,3))
cnt_vect.fit(text)
ftr_vect = cnt_vect.transform(text)
print(type(ftr_vect), ftr_vect.shape)
print(cnt_vect.vocabulary_)
```

    <class 'scipy.sparse.csr.csr_matrix'> (2, 201)
    {'the': 129, 'matrix': 77, 'is': 66, 'everywhere': 40, 'its': 74, 'all': 0, 'around': 11, 'us': 150, 'here': 51, 'even': 37, 'in': 59, 'this': 140, 'room': 106, 'you': 174, 'can': 25, 'see': 109, 'it': 69, 'out': 90, 'your': 193, 'window': 165, 'or': 83, 'on': 80, 'television': 126, 'feel': 43, 'when': 162, 'go': 46, 'to': 143, 'work': 171, 'church': 28, 'pay': 93, 'taxes': 125, 'the matrix': 132, 'matrix is': 78, 'is everywhere': 67, 'everywhere its': 41, 'its all': 75, 'all around': 1, 'around us': 12, 'us here': 151, 'here even': 52, 'even in': 38, 'in this': 60, 'this room': 141, 'room you': 107, 'you can': 177, 'can see': 26, 'see it': 110, 'it out': 70, 'out your': 91, 'your window': 199, 'window or': 166, 'or on': 86, 'on your': 81, 'your television': 197, 'television you': 127, 'you feel': 179, 'feel it': 44, 'it when': 72, 'when you': 163, 'you go': 181, 'go to': 47, 'to work': 148, 'work or': 172, 'or go': 84, 'to church': 146, 'church or': 29, 'or pay': 88, 'pay your': 94, 'your taxes': 196, 'the matrix is': 133, 'matrix is everywhere': 79, 'is everywhere its': 68, 'everywhere its all': 42, 'its all around': 76, 'all around us': 2, 'around us here': 13, 'us here even': 152, 'here even in': 53, 'even in this': 39, 'in this room': 61, 'this room you': 142, 'room you can': 108, 'you can see': 178, 'can see it': 27, 'see it out': 111, 'it out your': 71, 'out your window': 92, 'your window or': 200, 'window or on': 167, 'or on your': 87, 'on your television': 82, 'your television you': 198, 'television you feel': 128, 'you feel it': 180, 'feel it when': 45, 'it when you': 73, 'when you go': 164, 'you go to': 182, 'go to work': 49, 'to work or': 149, 'work or go': 173, 'or go to': 85, 'go to church': 48, 'to church or': 147, 'church or pay': 30, 'or pay your': 89, 'pay your taxes': 95, 'take': 121, 'blue': 22, 'pill': 96, 'and': 3, 'story': 118, 'ends': 34, 'wake': 153, 'bed': 14, 'believe': 17, 'whatever': 159, 'want': 156, 'red': 103, 'stay': 115, 'wonderland': 168, 'show': 112, 'how': 56, 'deep': 31, 'rabbit': 100, 'hole': 54, 'goes': 50, 'you take': 187, 'take the': 122, 'the blue': 130, 'blue pill': 23, 'pill and': 97, 'and the': 6, 'the story': 138, 'story ends': 119, 'ends you': 35, 'you wake': 189, 'wake in': 154, 'in your': 64, 'your bed': 194, 'bed and': 15, 'and you': 8, 'you believe': 175, 'believe whatever': 18, 'whatever you': 160, 'you want': 191, 'want to': 157, 'to believe': 144, 'believe you': 20, 'the red': 136, 'red pill': 104, 'you stay': 185, 'stay in': 116, 'in wonderland': 62, 'wonderland and': 169, 'and show': 4, 'show you': 113, 'you how': 183, 'how deep': 57, 'deep the': 32, 'the rabbit': 134, 'rabbit hole': 101, 'hole goes': 55, 'you take the': 188, 'take the blue': 123, 'the blue pill': 131, 'blue pill and': 24, 'pill and the': 98, 'and the story': 7, 'the story ends': 139, 'story ends you': 120, 'ends you wake': 36, 'you wake in': 190, 'wake in your': 155, 'in your bed': 65, 'your bed and': 195, 'bed and you': 16, 'and you believe': 9, 'you believe whatever': 176, 'believe whatever you': 19, 'whatever you want': 161, 'you want to': 192, 'want to believe': 158, 'to believe you': 145, 'believe you take': 21, 'take the red': 124, 'the red pill': 137, 'red pill and': 105, 'pill and you': 99, 'and you stay': 10, 'you stay in': 186, 'stay in wonderland': 117, 'in wonderland and': 63, 'wonderland and show': 170, 'and show you': 5, 'show you how': 114, 'you how deep': 184, 'how deep the': 58, 'deep the rabbit': 33, 'the rabbit hole': 135, 'rabbit hole goes': 102}
    



### 희소 행렬 - COO 형식


```python
import numpy as np

dense = np.array( [ [ 3, 0, 1 ], 
                    [0, 2, 0 ] ] )
```


```python
from scipy import sparse

# 0 이 아닌 데이터 추출
data = np.array([3,1,2])

# 행 위치와 열 위치를 각각 array로 생성 
row_pos = np.array([0,0,1])
col_pos = np.array([0,2,1])

# sparse 패키지의 coo_matrix를 이용하여 COO 형식으로 희소 행렬 생성
sparse_coo = sparse.coo_matrix((data, (row_pos,col_pos)))
```


```python
print(type(sparse_coo))
print(sparse_coo)
dense01=sparse_coo.toarray()
print(type(dense01),"\n", dense01)
```

    <class 'scipy.sparse.coo.coo_matrix'>
      (0, 0)	3
      (0, 2)	1
      (1, 1)	2
    <class 'numpy.ndarray'> 
     [[3 0 1]
     [0 2 0]]
    

### 희소 행렬 – CSR 형식


```python
from scipy import sparse

dense2 = np.array([[0,0,1,0,0,5],
             [1,4,0,3,2,5],
             [0,6,0,3,0,0],
             [2,0,0,0,0,0],
             [0,0,0,7,0,8],
             [1,0,0,0,0,0]])

# 0 이 아닌 데이터 추출
data2 = np.array([1, 5, 1, 4, 3, 2, 5, 6, 3, 2, 7, 8, 1])

# 행 위치와 열 위치를 각각 array로 생성 
row_pos = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 5])
col_pos = np.array([2, 5, 0, 1, 3, 4, 5, 1, 3, 0, 3, 5, 0])

# COO 형식으로 변환 
sparse_coo = sparse.coo_matrix((data2, (row_pos,col_pos)))

# 행 위치 배열의 고유한 값들의 시작 위치 인덱스를 배열로 생성
row_pos_ind = np.array([0, 2, 7, 9, 10, 12, 13])

# CSR 형식으로 변환 
sparse_csr = sparse.csr_matrix((data2, col_pos, row_pos_ind))

print('COO 변환된 데이터가 제대로 되었는지 다시 Dense로 출력 확인')
print(sparse_coo.toarray())
print('CSR 변환된 데이터가 제대로 되었는지 다시 Dense로 출력 확인')
print(sparse_csr.toarray())

```

    COO 변환된 데이터가 제대로 되었는지 다시 Dense로 출력 확인
    [[0 0 1 0 0 5]
     [1 4 0 3 2 5]
     [0 6 0 3 0 0]
     [2 0 0 0 0 0]
     [0 0 0 7 0 8]
     [1 0 0 0 0 0]]
    CSR 변환된 데이터가 제대로 되었는지 다시 Dense로 출력 확인
    [[0 0 1 0 0 5]
     [1 4 0 3 2 5]
     [0 6 0 3 0 0]
     [2 0 0 0 0 0]
     [0 0 0 7 0 8]
     [1 0 0 0 0 0]]
    


```python
print(sparse_csr)
```

      (0, 2)	1
      (0, 5)	5
      (1, 0)	1
      (1, 1)	4
      (1, 3)	3
      (1, 4)	2
      (1, 5)	5
      (2, 1)	6
      (2, 3)	3
      (3, 0)	2
      (4, 3)	7
      (4, 5)	8
      (5, 0)	1
    


```python
dense3 = np.array([[0,0,1,0,0,5],
             [1,4,0,3,2,5],
             [0,6,0,3,0,0],
             [2,0,0,0,0,0],
             [0,0,0,7,0,8],
             [1,0,0,0,0,0]])

coo = sparse.coo_matrix(dense3)
csr = sparse.csr_matrix(dense3)
```


```python
print(coo)
```

      (0, 2)	1
      (0, 5)	5
      (1, 0)	1
      (1, 1)	4
      (1, 3)	3
      (1, 4)	2
      (1, 5)	5
      (2, 1)	6
      (2, 3)	3
      (3, 0)	2
      (4, 3)	7
      (4, 5)	8
      (5, 0)	1
    


```python
print(csr)
```

      (0, 2)	1
      (0, 5)	5
      (1, 0)	1
      (1, 1)	4
      (1, 3)	3
      (1, 4)	2
      (1, 5)	5
      (2, 1)	6
      (2, 3)	3
      (3, 0)	2
      (4, 3)	7
      (4, 5)	8
      (5, 0)	1
    


```python

```
