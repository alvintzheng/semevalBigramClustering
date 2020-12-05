##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

import argparse
import sys
import types
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np
from HyperpartisanNewsReader import *
import time

def balance(X, Y):
    excessClass = 1
    oneCount = sum(Y)
    if oneCount < len(Y) / 2:
        excessClass = 0
    
    excessCount = 0
    for elem in Y:
        if elem == excessClass:
            excessCount += 1
    
    X = X.todense()
    excessCount -= len(Y) - excessCount
    for i in range(excessCount):
        for j in range(len(Y)):
            if Y[j] == excessClass:
                Y.pop(j)
                X = np.delete(X, j, 0)
                break
    X = sparse.lil_matrix(X)
    return X, Y

def do_experiment(args):
    vocab = HNVocab(args.vocabulary, args.vocab_size, args.stop_words, args.ngrams, args.cluster)
    # use word embeddings to group words together, make less bi-grams/tri-grams? get them from word-net?
    features = BagOfWordsFeatures(vocab, args.tag)
    labels = BinaryLabels()
    #classifier = MultinomialNB()
    #classifier = svm.SVC()
    classifier = LogisticRegression(max_iter=200)
    #classifier = DummyClassifier(strategy='stratified')
    #classifier = DummyClassifier(strategy='most_frequent')
    XTrain, XTrainIDs = features.process(args.training, args.train_size)
    YTrain = labels.process(args.labels, args.train_size)
    XTest, XTestIDs = features.process(args.test_data, args.test_size)
    YTest = labels.process(args.testLabels)
    XTest, YTest = balance(XTest, YTest)
    print(XTrain.shape)
    if not args.test_data is None:
        classifier.fit(XTrain, YTrain)
        #XTest, XTestIDs = features.process(args.test_data, args.test_size)
        #YTest = labels.process(args.testLabels)
        count = 0
        
        predictions = classifier.predict(XTest)
        scores = getScores(YTest, predictions)
        return scores


def writeOutput(IDs, predictions, confidences, labeler, file):
    for (i, p, c) in zip(IDs, predictions, confidences):
        prediction = labeler[p]
        confidence = c[p]
        elems = [str(i), str(prediction), str(confidence), "\n"]
        file.write(" ".join(elems))
    file.close()

def getScores(labels, predictions):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(len(predictions)):
        label = labels[i]
        prediction = predictions[i]
        if label == 1 and prediction == 1:
            TP += 1
        elif label == 0 and prediction == 0:
            TN += 1
        elif label == 1 and prediction == 0:
            FN += 1
        elif label == 0 and prediction == 1:
            FP += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    result = {'acc': accuracy, 'prec': precision, 'rec': recall, 'F1': F1}
    return result

'''
move command line to main.
add - n-grams (bigrams), type of word(lemma which should decrease feature count, pos, word), change model, and do normalization of different features.
'''

if __name__ == '__main__':
    #ignoring stop words for now
    vocabSize = 10000
    trainSize = 10000
    trainSet = 'publisher'
    testSet = 'article'
    ngrams = {'unigrams':True, 'bigrams':True}
    tag = 'spacy'
    cluster = True
    vocabSizeT = int(vocabSize/1000)
    trainSizeT = int(trainSize/1000)
    vocabularyFile = '{}clusters{}KV{}KA{}.txt'.format(tag, vocabSizeT, trainSizeT, trainSet)
    trainingFile = '/cs/cs159/data/semeval/articles-training-by{0}-20181122.parsed.xml'.format(trainSet)
    labelsFile = '/cs/cs159/data/semeval/ground-truth-training-by{0}-20181122.xml'.format(trainSet)
    testLabelsFile = '/cs/cs159/data/semeval/ground-truth-training-by{0}-20181122.xml'.format(testSet)
    testDataFile = '/cs/cs159/data/semeval/articles-training-by{0}-20181122.parsed.xml'.format(testSet)
    outputFile = 'out.txt'
    stopWords = 100
    testSize = None
    xValidate = 5
    args = types.SimpleNamespace()
    args.tag = tag
    args.training = open(trainingFile, 'rb')
    args.labels = open(labelsFile, 'rb')
    args.testLabels = open(testLabelsFile, 'rb')
    args.vocabulary = open(vocabularyFile, 'r')
    args.output_file = open(outputFile, 'w')
    args.stop_words = stopWords
    args.vocab_size = vocabSize
    args.train_size = trainSize
    args.test_size = testSize
    args.test_data = open(testDataFile, 'rb')
    args.xvalidate = xValidate
    args.ngrams = ngrams
    args.cluster = cluster

    startTime = time.perf_counter()
    scores = do_experiment(args)
    endTime = time.perf_counter()
    timeTaken = endTime - startTime
    timesFile = open('times.txt', 'a')
    newline = 'performance{}KV{}TR{}TS{}KTrS{}U{}B{}tag: \t{} time\t{} acc'.format(vocabSizeT, trainSet, testSet, trainSizeT, ngrams['unigrams'], ngrams['bigrams'], tag, round(timeTaken, 3), scores)
    timesFile.write(newline + "\n")
    for fp in (args.output_file, args.training, args.labels, args.testLabels, args.vocabulary, timesFile): fp.close()