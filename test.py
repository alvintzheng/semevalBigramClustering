from gensim.models import Word2Vec
from sklearn import cluster
from sklearn import metrics

sentences = [['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
            ['this', 'is',  'another', 'book'],
            ['one', 'more', 'book'],
            ['this', 'is', 'the', 'new', 'post'],
                        ['this', 'is', 'about', 'machine', 'learning', 'post'],  
            ['and', 'this', 'is', 'the', 'last', 'post']]
model = Word2Vec(sentences, min_count=1)
model.save('sampleModel.model')
model = Word2Vec.load('sampleModel.model')
print (model.wv.similarity('this', 'is'))
print (model.wv.similarity('post', 'book'))
print (model.wv.most_similar(positive=['machine'], negative=[], topn=2))

X = []
Y = []
for i in (sentences[0]):
    x = model.wv[i]
    X.append(x)
    Y.append(i)


kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
print(labels)
print(Y)