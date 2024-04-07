# Word2Vec CBOW And PCA To 2 Dim

Training the Word2Vec CBOW model with articles to convert keywords to vectors, reducing to 2 dimensions using PCA, determining 5 clusters with KMeans, and achieving vector visualization.

## Overview

- Language: Python v3.9.15
- Package: gensim v4.1.2
- Model: Word2vec CBOW

## Requirement

- article to train Word2vec
- dictionary to cut word from article

## Run

### train model

```
python cbow-word2vec.py
```

### use model

```
python keyword-vector.py
```


## Example Of Result

### PCA 

![image](https://github.com/yuhexiong/cbow-word2vec-pca-nlp-python/blob/main/image/pca.png)