---
title: 영화 리뷰 분석 프로젝트
tags: Review_Analysis_Project
typora-root-url: ../
---

# 프로젝트 소개



## 배경:

- 기존의 영화리뷰 사이트는 사용자가 매긴 리뷰 평점을 통계적으로 종합한 정보만 제공
- 해당 정보만으로는 전체 영화 리뷰 정보에 대해 파악 불가능
- 영화 시청을 결정하는데 있어 중요한 정보를 제공 받지 못하는 상황



## 목표:

- 영화 리뷰를 분석해 해당 리뷰의 긍정/부정 판단

- 리뷰 키워드 추출 및 클러스터링을 통해 전반적인 여론 파악

- LLM모델 활용해 영화 리뷰에 대한 전체적인 요약 정보 제공

  

## 대상:

- 아직 시청하지 않은 영화에 대한 평가가 궁금한 사람

  

## 효과:

- 영화의 전반적인 여론을 알려주는 지표를 통해 해당 영화 평가 확인 가능
- 영화 리뷰에 대한 요약된 정보를 바탕으로 영화의 어떠한 부분이 긍정적이고 부정적인지에 대한 정보를 제공 받을 수 있음



## 사용된 학습 데이터셋:

- 한국 영화 리뷰 평점 데이터셋: [https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt](https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt)
- 네이버, 다음 영화 리뷰 크롤링



## 과정:

- 데이터 전처리
  - Tf-idf
  - word2vec
  - BERT Tokenizer
- 감정 분석 모델 학습
  - Linear Regression
  - Random Forest
  - XGBoost
  - LSTM
  - BERT pretrained model
- 리뷰 클러스터링
  - KMeans
  - 추후 추가 예정
- 리뷰 키워드 추출
  - Tf-idf
  - TextRank
  - Cosine Similarity
- LLM 활용해 요약된 정보 제공
  - ChatGPT
