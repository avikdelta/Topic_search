import ConfigParser
import os
import numpy as np
import scipy
import scipy.io
import urllib2
import socket
import ssl

#---------------------------------------------------------------
# Splits word by document matrix into train, val, and test sets
#---------------------------------------------------------------
def splitData(corpusMatFile,config):

    # Load word by document matrix
    M = scipy.io.loadmat(corpusMatFile)['M']

    # Get number of documents in corpus
    M = M.tocsc()
    numWords = M.shape[0]
    numDocs = M.shape[1]
    print 'Number of words = ' + str(numWords)
    print 'Number of documents = ' + str(numDocs)

    # Train-Val-Test split
    fraction = [config['trainFrac'],config['valFrac'],config['testFrac']]
    filterFile = config['filterFile']
    fw = open(filterFile,'w')
    tot = sum(fraction)
    trainMin = 0
    trainMax = fraction[0]/float(tot)
    valMin = fraction[0]/float(tot)
    valMax = (fraction[0]+fraction[1])/float(tot)
    testMin = (fraction[0]+fraction[1])/float(tot)
    testMax = (fraction[0]+fraction[1]+fraction[2])/float(tot)
    for i in range(numDocs):
        r = np.random.rand()
        if r>=trainMin and r<trainMax:
            fw.write("1 ")
        elif r>=valMin and r<valMax:
            fw.write("2 ")
        else:
            fw.write("3 ")

    fw.close()
    print "Split complete ! Filter file written."
    
    return

#---------------------------------------------------------------
# Reads filter file with train, val, and test splits
#---------------------------------------------------------------
def readFilter(config):

    filterFile = config['filterFile']
    filterList = [[],[],[]]
    print 'Reading filters ...'
    with open(filterFile,'r') as fr:
        line = fr.readline()
        line = line.strip().split()
        for i in range(len(line)):
            if line[i]=="1":
                filterList[0].append(i)
            elif line[i]=="2":
                filterList[1].append(i)
            else:
                filterList[2].append(i)

    return filterList

#---------------------------------------------------------------
# Gets word index from dictionary
#---------------------------------------------------------------
def getWordIndex(dictFile, word):

    fr = open(dictFile,'r')
    index = 0
    for line in fr:

        if line.strip() == word:
            print 'Word found! Index = ' + str(index)
            fr.close()
            return index
        else:
            index += 1

    fr.close()
    print 'Word not found!'
    return -1

#---------------------------------------------------------------
# Gets word in dictionary given an index
#---------------------------------------------------------------
def getWord(dictFile, index):

    fr = open(dictFile,'r')
    count = 0
    for line in fr:

        if count == index:
            word = line.strip()
            fr.close()
            return word
        else:
            count += 1

    fr.close()
    print 'Index exceeds dictionary size!'
    return ''

#---------------------------------------------------------------
# Saves top N words in the topic retuned by search by whiten
# algorithm
#---------------------------------------------------------------
def saveTopWords(dictFile, N, mu, labelWord):

    #Sort in descending order
    sortLikelihoodIdx = np.argsort(-mu)

    print 'Printing top ' + str(N) + ' words from the topic ...'
    print '------------------------'
    topCount = 0
    dirName = 'res'
    fileName = labelWord + '_topwords_whiten.txt'
    filePath = os.path.join(dirName,fileName)
    fw = open(filePath,'w')
    top10List = []
    for i in range(len(mu)):

        if i < N:
            topWord = getWord(dictFile,sortLikelihoodIdx[i])
            if i<10:
                if len(topWord)>4:
                    # Removes zzz_ from start of word (zzz_ is used for tokening proper nouns)
                    if topWord[:4]=="zzz_":
                        wordnzzz = topWord[4:]
                        top10List.append(wordnzzz)
                        topWord = wordnzzz
                    else:
                        top10List.append(topWord)
                else:
                    top10List.append(topWord)
                
            print str(i+1) + '<< ' + topWord + ' >> prob = ' + str(mu[sortLikelihoodIdx[i]])
            fw.write(topWord + ' ' + str(mu[sortLikelihoodIdx[i]]) + '\n')
            
        if mu[sortLikelihoodIdx[i]] > 0:
            topCount += 1

    fw.close()
    print 'Result saved !'    

    return top10List

#---------------------------------------------------------------
# Saves top N words in the topic retuned by search by NMF
# algorithm
#---------------------------------------------------------------
def saveTopWordsNMF(dictFile, N, mu, labelWord):

    #Sort in descending order
    sortLikelihoodIdx = np.argsort(-mu)

    print 'Printing top ' + str(N) + ' words from the topic ...'
    print '------------------------'
    topCount = 0
    dirName = 'res'
    fileName = labelWord + '_topwords_NMF.txt'
    filePath = os.path.join(dirName,fileName)
    fw = open(filePath,'w')
    top10List = []
    for i in range(len(mu)):

        if i < N:
            topWord = getWord(dictFile,sortLikelihoodIdx[i])
            if i<10:
                if len(topWord)>4:
                    # Removes zzz_ from start of word (zzz_ is used for tokening proper nouns)
                    if topWord[:4]=="zzz_":
                        wordnzzz = topWord[4:]
                        top10List.append(wordnzzz)
                        topWord = wordnzzz
                    else:
                        top10List.append(topWord)
                else:
                    top10List.append(topWord)
                
            print str(i+1) + '<< ' + topWord + ' >> prob = ' + str(mu[sortLikelihoodIdx[i]])
            fw.write(topWord + ' ' + str(mu[sortLikelihoodIdx[i]]) + '\n')
            
        if mu[sortLikelihoodIdx[i]] > 0:
            topCount += 1

    fw.close()
    print 'Result saved !'    

    return top10List

#---------------------------------------------------------------
# Saves top N words in the topic retuned by search by SS-NMF
# algorithm
#---------------------------------------------------------------
def saveTopWordsSSNMF(dictFile, N, mu, labelWord):

    #Sort in descending order
    sortLikelihoodIdx = np.argsort(-mu)

    print 'Printing top ' + str(N) + ' words from the topic ...'
    print '------------------------'
    topCount = 0
    dirName = 'res'
    fileName = labelWord + '_topwords_SSNMF.txt'
    filePath = os.path.join(dirName,fileName)
    fw = open(filePath,'w')
    top10List = []
    for i in range(len(mu)):

        if i < N:
            topWord = getWord(dictFile,sortLikelihoodIdx[i])
            if i<10:
                if len(topWord)>4:
                    # Removes zzz_ from start of word (zzz_ is used for tokening proper nouns)
                    if topWord[:4]=="zzz_":
                        wordnzzz = topWord[4:]
                        top10List.append(wordnzzz)
                        topWord = wordnzzz
                    else:
                        top10List.append(topWord)
                else:
                    top10List.append(topWord)
                
            print str(i+1) + '<< ' + topWord + ' >> prob = ' + str(mu[sortLikelihoodIdx[i]])
            fw.write(topWord + ' ' + str(mu[sortLikelihoodIdx[i]]) + '\n')
            
        if mu[sortLikelihoodIdx[i]] > 0:
            topCount += 1

    fw.close()
    print 'Result saved !'    

    return top10List


#---------------------------------------------------------------
# Computes PMI score for search by Whitening algorithm based on
# top 20 words in the topic
#---------------------------------------------------------------
def computePMIWhiten(labelWord):

    dirName = 'res'
    fileName = labelWord + '_topwords_whiten.txt'
    filePath = os.path.join(dirName,fileName)
    fr = open(filePath,'r')
    pmi = 0.0
    count = 0
    print 'Computing top 20 pmi ...'
    for i in range(20):
        line = fr.readline()
        line = line.strip().split()
        word1 = labelWord
        word2 = line[0]
        query = "http://palmetto.aksw.org/palmetto-webapp/service/npmi?words="+word1+"%20"+word2
        req = urllib2.Request(query)
        try:
            timeoutTimeSec = 5
            response = urllib2.urlopen(req, timeout = timeoutTimeSec)
            pmi += float(response.read())
            count += 1
            response.close()
        except (ssl.SSLError, urllib2.URLError, socket.timeout, socket.error), e:
            print 'Timeout Error! Could not connect to palmetto !'
            print e

    fr.close()
    if count>0:
        pmi = pmi/float(count)
    else:
        pmi = 0
        
    return pmi

#---------------------------------------------------------------
# Computes PMI score for search by NMF algorithm based on
# top 20 words in the topic
#---------------------------------------------------------------
def computePMINMF(labelWord):

    dirName = 'res'
    fileName = labelWord + '_topwords_NMF.txt'
    filePath = os.path.join(dirName,fileName)
    fr = open(filePath,'r')
    pmi = 0.0
    count = 0
    print 'Computing top 20 pmi ...'
    for i in range(20):
        line = fr.readline()
        line = line.strip().split()
        word1 = labelWord
        word2 = line[0]
        query = "http://palmetto.aksw.org/palmetto-webapp/service/npmi?words="+word1+"%20"+word2
        req = urllib2.Request(query)
        try:
            timeoutTimeSec = 5
            response = urllib2.urlopen(req, timeout = timeoutTimeSec)
            pmi += float(response.read())
            count += 1
        except (ssl.SSLError, urllib2.URLError, socket.timeout, socket.error), e:
            print 'Timeout Error! Could not connect to palmetto !'
            print e

    fr.close()
    if count>0:
        pmi = pmi/float(count)
    else:
        pmi = 0
        
    return pmi

#---------------------------------------------------------------
# Computes PMI score for search by SS-NMF algorithm based on
# top 20 words in the topic
#---------------------------------------------------------------
def computePMISSNMF(labelWord):

    dirName = 'res'
    fileName = labelWord + '_topwords_SSNMF.txt'
    filePath = os.path.join(dirName,fileName)
    fr = open(filePath,'r')
    pmi = 0.0
    count = 0
    print 'Computing top 20 pmi ...'
    for i in range(20):
        line = fr.readline()
        line = line.strip().split()
        word1 = labelWord
        word2 = line[0]
        query = "http://palmetto.aksw.org/palmetto-webapp/service/npmi?words="+word1+"%20"+word2
        req = urllib2.Request(query)
        try:
            timeoutTimeSec = 5
            response = urllib2.urlopen(req, timeout = timeoutTimeSec)
            pmi += float(response.read())
            count += 1
        except (ssl.SSLError, urllib2.URLError, socket.timeout, socket.error), e:
            print 'Timeout Error! Could not connect to palmetto !'
            print e

    fr.close()
    if count>0:
        pmi = pmi/float(count)
    else:
        pmi = 0
        
    return pmi    

#---------------------------------------------------------------
# Load config class from config file
#---------------------------------------------------------------    
def loadConfig(configFile):

    #configFile = "configCorpus.txt"
    cfg = ConfigParser.ConfigParser()
    try:
        cfg.read(configFile)
        # Load configs   
        config = {}
        config['corpusFile'] = cfg.get("SETTING","corpus_file")
        config['dictFile'] = cfg.get("SETTING","dictionary_file")
        config['matFile'] = cfg.get("SETTING","mat_file_name")
        config['filterFile'] = cfg.get("SETTING","filter_file")
        config['stopwordsFile'] = cfg.get("SETTING","stopwords_file")
        config['trainFrac'] = cfg.getfloat("SETTING","train_fraction")
        config['valFrac'] = cfg.getfloat("SETTING","validation_fraction")
        config['testFrac'] = cfg.getfloat("SETTING","test_fraction")
        config['rareWordTh'] = cfg.getint("SETTING","rare_word_threshold")
        config['K'] = cfg.getint("SETTING","num_topics")
        config['N'] = cfg.getint("SETTING","top_words_to_display")
        config['alpha0'] = cfg.getfloat("SETTING","alpha_0")
        config['weight'] = cfg.getfloat("SETTING","weight")

        return config

    except (ConfigParser.Error,ConfigParser.NoSectionError), e:
        print 'Error loading config file.'
        print e
        return {}

#---------------------------------------------------------------
# Preprocess dataset.
# 1) Converts bag of words to sparse matrix
# 2) Removes rare and stop words to truncate the vocabulary
#---------------------------------------------------------------    
def preprocess(config):

    matFile = config['matFile']
    corpusFile = config['corpusFile']
    rareWordTh = config['rareWordTh']
    
    print 'Converting corpus to mat file ...'
    cmdStr = 'python uci_to_scipy.py ' + corpusFile + ' ' + matFile
    os.system(cmdStr)

    print 'Copying stopwords file ...'
    stopwordsFile = config['stopwordsFile']
    fw = open('stopwords.txt','w')
    fr = open(stopwordsFile,'r')
    for line in fr:
        fw.write(line)
    fr.close()
    fw.close()
    
    print 'Removing rare words ...'
    rareWordTh = config['rareWordTh']
    dictFile = config['dictFile']
    cmdStr = 'python truncate_vocabulary.py ' + matFile + ' ' + dictFile + ' ' + str(rareWordTh)
    os.system(cmdStr)

    truncMatFile = matFile + '.trunc'
    truncDictFile = dictFile + '.trunc'

    return (truncMatFile, truncDictFile)
