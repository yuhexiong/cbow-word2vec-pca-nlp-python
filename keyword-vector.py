import numpy as np
import pandas as pd
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as font

model = word2vec.Word2Vec.load('word2vec_new.model')

# keyword as example
keywords = ['C', 'C#', 'C++', 'Django', 'Python', '機器學習', '深度學習', '演算法']

newKeywords = []
for word in keywords:
    word = word.replace('-', ' ')
    word = word.replace('/', ' ')
    word = word.lower()
    newKeywords.append(word)

# convert to vector
words = []
wordVectors = []

for kw in newKeywords:
    try:
        wordVectors.append(np.array(model.wv[kw]))
        words.append(kw)
    except:
        print(kw, 'not in model')
twoDimVectors = PCA().fit_transform(wordVectors)[:,:2]

# kmeans to decide the number of group
silhouette_avg = []
for i in range(2,11):
    kmeans_fit = KMeans(n_clusters = i).fit(wordVectors)
    silhouette_avg.append(silhouette_score(wordVectors, kmeans_fit.labels_))
plt.plot(range(2,11), silhouette_avg)
plt.show()

# use 5 groups
kmeans = KMeans(n_clusters = 5)
kmeans.fit(wordVectors) 
label = kmeans.predict(wordVectors)

# pca visualization
font1 = font(fname="/font/NotoSansTC-Regular.otf")

plt.figure(figsize=(15,15))
plt.xlabel('x')
plt.ylabel('y')

plt.scatter(
    twoDimVectors[:,0], twoDimVectors[:,1], 
    c = label,         # Specify the color for each point
    edgecolor = 'k',   # No edge color
    alpha = 0.6        # Opacity level
)
for word, (x,y) in zip(words, twoDimVectors):
    plt.text(x+0.05, y+0.05, word, fontproperties=font1)

plt.show() 

# save vector as csv
keywords2DimVector = pd.DataFrame(columns = ['key', 'x', 'y', 'cluster'])
for i in range(len(twoDimVectors)):
    keywords2DimVector.loc[i] = [words[i], twoDimVectors[i,0], twoDimVectors[i,1], label[i]]
keywords2DimVector.to_csv('keywords2DimVector.csv')