from itertools import islice
from html import unescape
from scipy import sparse 
from lxml import etree
from gensim.models import Word2Vec
from sklearn import cluster
from HyperpartisanNewsReader import *
import time

def getSentences(fileName, tag):
    data_file = open(fileName, 'rb')
    articles = do_xml_parse(data_file, 'article')
    for article in articles:
        for spacyElem in article.iterfind(tag):
            words = spacyElem.text.split(" ")
            yield words

class sentenceIterable(object):
    def __init__(self, fileName, tag, vocab, articleCount, stopWords):
        self.vocab = vocab[stopWords:]
        self.fileName = fileName
        self.tag = tag
        self.dict = dict([(w, i) for (i, w) in enumerate(self.vocab)])
        self.articleCount = articleCount
        self.types = []
    def __iter__(self):
        data_file = open(self.fileName, 'rb')
        articles = do_xml_parse(data_file, 'article', progress_message="Article {}", max_elements=self.articleCount)
        for article in articles:
            for spacyElem in article.iterfind(self.tag):
                words = spacyElem.text.split(" ")
                result = [word for word in words if (word in self.dict)]
                yield result

if __name__ == '__main__':
    articleCount = 600000
    tag = 'spacy'
    dataset = 'publisher'
    vocabFile = open('/cs/cs159/data/semeval/vocab.txt', 'r')
    #vocabFile = open('./posVocab.txt', 'r')
    vocabSize = 10000
    vocabSizeT = int(vocabSize/1000)
    articleCountT = int(articleCount/1000)
    
    stopWords = 100 if not tag == 'tag' else 0
    startTime = time.perf_counter()
    vocab = [w.strip() for w in islice(vocabFile, 0, vocabSize)]
    
    dataFileName = '/cs/cs159/data/semeval/articles-training-by{0}-20181122.parsed.xml'.format(dataset)

    modelFile = "{}embeddings{}KV{}KA{}.model".format(tag, vocabSizeT, articleCountT, dataset)
    model = Word2Vec(sentenceIterable(dataFileName, tag, vocab, articleCount, stopWords), min_count=1)
    model.save(modelFile)

    model = Word2Vec.load(modelFile)
    Y = [x for x in vocab if x in model.wv]
    X = [model.wv[x] for x in Y]
    count = len(Y)
    clusterCount = int(count ** (1/2))
    kmeans = cluster.KMeans(n_clusters=clusterCount)
    kmeans.fit(X)
    labels = kmeans.labels_
    
    outFile = open('./{}clusters{}KV{}KA{}.txt'.format(tag, vocabSizeT, articleCountT, dataset), 'w')
    outFile.write(str(clusterCount) + "\n")
    for i in range(count):
        newLine = Y[i] + " " + str(labels[i]) + "\n"
        outFile.write(newLine)
    outFile.close()

    endTime = time.perf_counter()
    timeTaken = endTime - startTime
    timesFile = open('times.txt', 'a')
    newline = '{}clustering{}KV{}DS{}DsS: \t{} time'.format(tag, vocabSizeT, dataset, vocabSize, round(timeTaken, 3))
    timesFile.write(newline + "\n")
    timesFile.close()



