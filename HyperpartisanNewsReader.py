##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

from abc import ABC, abstractmethod
from itertools import islice
from html import unescape
from scipy import sparse 
from lxml import etree
import sys

from collections import Counter

#####################################################################
# HELPER FUNCTIONS
#####################################################################

def do_xml_parse(fp, tag, max_elements=None, progress_message=None):
    """ 
    Parses cleaned up spacy-processed XML files
    """
    fp.seek(0)

    # islice is just like array slicing but for iterators
    elements = enumerate(islice(etree.iterparse(fp, tag=tag), max_elements))
    for i, (event, elem) in elements:
        yield elem #return one at a time from this method, as an iterator
        elem.clear() #clear content of element we already returned to save memory
        if progress_message and (i % 1000 == 0): 
            print(progress_message.format(i), file=sys.stderr, end='\r')
    if progress_message: print(file=sys.stderr)

def short_xml_parse(fp, tag, max_elements=None): 
    """ 
    Parses cleaned up spacy-processed XML files (but not very well)
    """
    elements = etree.parse(fp).findall(tag)
    N = max_elements if max_elements is not None else len(elements)
    return elements[:N] #returning all at once might be slow/much memory?

#####################################################################
# HNVocab
#####################################################################

class HNVocab(object): 
    def __init__(self, vocab_file, vocab_size, num_stop_words, usages, cluster): 
        #change: now vocab file is vocab + clusters
        self.vocab = []
        self.ngrams = []
        self.clusters = []
        self.cluster = cluster
        self.usages = usages
        unigrams = []
        clusterCount = int(vocab_file.readline())
        for line in vocab_file:
            elems = line.strip().split(" ")
            word = elems[0]
            cluster = elems[1]
            unigrams.append(word)
            self.clusters.append(cluster)

        self.vocab = unigrams
        #Bigrams
        bigrams = []
        if self.cluster:
            for i in range(clusterCount):
                for j in range(clusterCount):
                    bigram = "class{0}-class{1}".format(i, j)
                    bigrams.append(bigram)
        else:
            for gram1 in unigrams:
                for gram2 in unigrams:
                    bigram = "{} {}".format(gram1, gram2)
                    bigrams.append(bigram)

        if (usages['unigrams']):
            self.ngrams += unigrams
        if (usages['bigrams']):
            self.ngrams += bigrams
        
        self.dict = dict([(w, i) for (i, w) in enumerate(self.ngrams)])
        self.vocab_dict = dict([(w, i) for (i, w) in enumerate(self.vocab)])

    def getBigram(self, word1, word2):
        if not word1 in self.vocab_dict or not word2 in self.vocab_dict:
                return ""
        if self.cluster:
            class1 = self.clusters[self.vocab_dict[word1]]
            class2 = self.clusters[self.vocab_dict[word2]]
            return "class{0}-class{1}".format(class1, class2)
        else: 
            return "{} {}".format(word1, word2)

    def __len__(self): 
        return len(self.dict)

    def index_to_label(self, i): 
        return self.ngrams[i]

    def __getitem__(self, key):
        if key in self.dict: return self.dict[key]
        else: return None

#####################################################################
# HNLabels
#####################################################################

class HNLabels(ABC): #extends the abstract base class which probably allows abstract methods
    def __init__(self): 
        '''
        'Initialiation', Not sure if these lines are necessary
        '''
        self.labels = None
        self._label_list = None

    def __getitem__(self, index):
        """ return the label at this index """
        return self._label_list[index]

    def process(self, label_file, max_instances=None):
        '''
        Obtain labels in a list and dict similar to vocab. In order for article labels.
        '''
        articles = do_xml_parse(label_file, 'article', max_elements=max_instances)
        y_labeled = list(map(self._extract_label, articles))
        if self.labels is None:
            self._label_list = sorted(set(y_labeled))
            self.labels = dict([(x,i) for (i,x) in enumerate(self._label_list)])
            
        y = [self.labels[x] for x in y_labeled]
        return y

    @abstractmethod     # just for show, shouldnt be used but defines some signature needed to be included in subclasses
    def _extract_label(self, article):
        """ Return the label for this article """
        return "Unknown"

#####################################################################
# HNFeatures
#####################################################################

class HNFeatures(ABC): #extends the abstract base class which probably allows abstract methods
    def __init__(self, vocab, tag):
        self.vocab = vocab
        self.usages = vocab.usages
        self.tag = tag

    def extract_text(self, article):
        '''
        get contents of spacy tags and split overall into tokens
        '''
        return unescape("".join([x for x in article.find(self.tag).itertext()]).lower()).split()

    def process(self, data_file, max_instances=None):
        '''
        get feature matrix for articles and id list
        '''
        if max_instances == None:
            N = len([1 for article in do_xml_parse(data_file, 'article')])
        else:
            N = max_instances

        # apparently allows you to add new rows whenever but I think here we don't ever add more
        X = sparse.lil_matrix((N, self._get_num_features()), dtype='uint8')
        
        ids = []
        articles = do_xml_parse(data_file, 'article', 
            max_elements=N, progress_message="Article {}")
        for i, article in enumerate(articles):
            ids.append(article.get("id"))
            for j, value in self._extract_features(article):
                # setting feature j in article i to value
                X[i,j] = value
        return X, ids

    @abstractmethod
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return "Unknown"

    @abstractmethod            
    def _extract_features(self, article):
        """ Returns a list of the features in the article """
        return []

    @abstractmethod        
    def _get_num_features(self):
        """ Return the total number of features """
        return -1

#####################################################################

#####################################################################
# BinaryLabels
#####################################################################

class BinaryLabels(HNLabels):

    def _extract_label(self, article):
        """ Return the label for this article """
        return article.get('hyperpartisan')

#####################################################################
# BagOfWordsFeatures
#####################################################################

class BagOfWordsFeatures(HNFeatures):
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return self.vocab.ngrams[i]
          
    def _extract_features(self, article):
        """ Returns a list of the features in the article """
        counts = Counter()
        for spacyElem in article.iterfind(self.tag):
            words = spacyElem.text.split(" ")
            if self.usages['unigrams']:
                for word in words:
                    if word in self.vocab.vocab_dict:
                        counts[self.vocab.dict[word]] += 1

            #bigram stuff
            if self.usages['bigrams']:
                for i in range(len(words) - 1):
                    word1 = words[i]
                    word2 = words[i + 1]
                    #bigram = words[i] + " " + words[i+1]
                    bigram = self.vocab.getBigram(word1, word2)
                    if bigram in self.vocab.dict:
                        pass
                        counts[self.vocab.dict[bigram]] += 1
        return counts.most_common()
      
    def _get_num_features(self):
        """ Return the total number of features """
        return len(self.vocab.ngrams)