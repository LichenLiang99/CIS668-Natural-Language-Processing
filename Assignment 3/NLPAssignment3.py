#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
import re
from nltk import tokenize
from nltk import FreqDist
from nltk.collocations import *
from nltk.corpus import stopwords
from nltk.corpus import sentence_polarity
import random


# In[3]:


#open the file, split lines and obtain the line starting with reviewText.
#save this line to a new text file called new.txt and open it

f = open("baby.txt")
text = f.read()
text_split = text.splitlines()

def reviewTexts(f):
    f2 = open("new.txt",'w+')
    for rt in f:
        if 'reviewText' in rt:
            writeText = re.sub('reviewText:', '', rt)
            f2.write(writeText+'\n')
    f2.close()
    
reviewTexts(text_split)

f3 = open("new.txt")
text2 = f3.read()
text2[0:1000]


# In[4]:


#tokenize the sentences and change all to lower case

tokenize_sentence = tokenize.sent_tokenize(text2)
tokenize_sentence = [w.lower( ) for w in tokenize_sentence]
print(len(tokenize_sentence))
print(tokenize_sentence[:20])


# In[5]:


#get the sentences we previously used to analysing movie reviews and shuffle it

sentences = sentence_polarity.sents()
documents = [(sent, cat) for cat in sentence_polarity.categories() for sent in sentence_polarity.sents(categories=cat)]
random.shuffle(documents)
documents[:20]


# In[6]:


#get all the words, make sure they are words and change all to lower case

all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)


# In[7]:


all_words_list = [w.lower( ) for w in all_words_list]
all_words_list = [w for w in all_words_list if w.isalpha()]
print(len(all_words_list))
print(all_words_list[:200])


# In[8]:


#get stopwords and negation words

stopwords = nltk.corpus.stopwords.words('english')
morestopwords = ['could','would','might','must','need','sha','wo','y',"'s","'d","'ll","'t","'m","'re","'ve"]
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
negationwords.extend(['ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])
stopwords = stopwords + morestopwords

#stopwords for NOT feature should not contain negation words
stopwords2 = [w for w in stopwords if w not in negationwords]


# In[9]:


#for SL features, remove all the stopwords first
#calculate the classifier accuracy using Naive Bayes Classifier

all_words = [w for w in all_words_list if not w in stopwords]
all_words = nltk.FreqDist(all_words_list)
word_item = all_words.most_common(2000)
word_features = [word for (word,count) in word_item]
print(word_features[:50])


# In[10]:


SLpath = 'subjclueslen1-HLTEMNLP05.tff'
def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict
SL = readSubjectivity(SLpath)


# In[11]:


def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)      
    return features

SL_featuresets = [(SL_features(d, word_features, SL), c) for (d, c) in documents]
train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)


# In[12]:


#for NOT feature, we want to make sure the stopwords do not contain negation words

all_words2 = [w for w in all_words_list if not w in stopwords2]
all_words2 = nltk.FreqDist(all_words2)
word_items2 = all_words2.most_common(2000)
word_features2 = [word for (word,count) in word_items2]
def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    return features

NOT_featuresets = [(NOT_features(d, word_features2, negationwords), c) for (d, c) in documents]
train_set, test_set = NOT_featuresets[1000:], NOT_featuresets[:1000]
classifier2 = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier2, test_set)


# In[13]:


#loop through each sentence, decide whether it's positive or negative,
#and save this sentence to the list and its corresponding file.
#Since it takes too long due to large amount of data, I have manually stopped it and 
#only analyse the sentences categorized in the text files

SLpositive = []
SLnegative = []
SLp = open('SLpositive.txt', 'w')
SLn = open('SLnegative.txt', 'w')

for sentence in tokenize_sentence:
    word = nltk.word_tokenize(sentence)
    feature = SL_features(word, word_features, SL)
    if classifier.classify(feature) == 'pos':
        SLpositive.append(sentence)
        SLp.write(sentence + '\n')
    elif classifier.classify(feature) == 'neg':
        SLnegative.append(sentence)
        SLn.write(sentence + '\n')
        
SLp.close()
SLn.close()



# In[14]:


print(len(SLpositive))
print(SLpositive[:10])

print(len(SLnegative))
print(SLnegative[:10])


# In[15]:


#loop through each sentence, decide whether it's positive or negative,
#and save this sentence to the list and its corresponding file.
#Since it takes too long due to large amount of data, I have manually stopped it and 
#only analyse the sentences categorized in the text files

NOTpositive = []
NOTnegative = []
NOTp = open('NOTpositive.txt', 'w')
NOTn = open('NOTnegative.txt', 'w')

for sentence in tokenize_sentence:
    word = nltk.word_tokenize(sentence)
    feature = NOT_features(word, word_features2, negationwords)
    if classifier2.classify(feature) == 'pos':
        NOTpositive.append(sentence)
        NOTp.write(sentence + '\n')
    elif classifier2.classify(feature) == 'neg':
        NOTnegative.append(sentence)
        NOTn.write(sentence + '\n')

NOTp.close()
NOTn.close()





# In[16]:


print(len(NOTpositive))
print(NOTpositive[:10])

print(len(NOTnegative))
print(NOTnegative[:10])


# In[ ]:




