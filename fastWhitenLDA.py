import numpy as np
import numpy.matlib
import scipy
import scipy.io
from scipy.sparse.linalg import svds
from utils import *
import sys
import time
import copy
import os

#---------------------------------------------------------------
# Apply filter to word by document matrix
#---------------------------------------------------------------
def applyFilter(corpusMat,filterList):

    corpusMat = corpusMat.tocsc()
    corpusMat = corpusMat[:,filterList]

    return corpusMat.tocsr()

#---------------------------------------------------------------
# Main search by Whitening algo function
#---------------------------------------------------------------    
def searchByWhiteningLDA(config, labelWord, corpusMatFile, dictFile):

    # Get word index from dictionary
    widx = getWordIndex(dictFile, labelWord)

    if widx < 0:
        print 'Error: Label word not found in dictionary'
        return

    # Load word by document matrix
    M = scipy.io.loadmat(corpusMatFile)['M']

    # Make column normalized M and weight vector W
    M = M.tocsc()
    newdata = np.zeros(M.data.shape[0], dtype=np.float64)
    print 'Normalizing columns and computing weight vector ...'
    for i in xrange(M.indptr.size-1):

        begin = M.indptr[i]
        end = M.indptr[i+1]
        totWord = np.sum(M.data[begin:end])
        if totWord > 0:
            newdata[begin:end] = np.divide(M.data[begin:end],float(totWord))


    M.data = newdata

    # Compute training, validation, test sets
    filterList = readFilter(config)
    Mtrain = M[:,filterList[0]]
    Mval = M[:,filterList[1]]
    Mtest = M[:,filterList[2]]
    numWords = Mtrain.shape[0]
    numDocs = Mtrain.shape[1]
    print 'Number of training words = ' + str(numWords)
    print 'Number of training documents = ' + str(numDocs)
    print 'Number of validation documents =', Mval.shape[1]
    print 'Number of test documents =', Mtest.shape[1]

    # Consider training set for remaining algo
    M = Mtrain
    alpha0 = config['alpha0']
    
    # Compute x and mean m
    print 'Computing vector x ...'
    M = M.tocsr()
    m = np.zeros(numWords,dtype=np.float64)
    for i in xrange(M.indptr.size-1):

        begin = M.indptr[i]
        end = M.indptr[i+1]        
        m[i] = np.sum(M.data[begin:end])/float(numDocs)

    x = np.zeros(numWords,dtype=np.float64)
    x = alpha0*m

    # Compute A
    print 'Computing matrix A ...'
    T2 = M*M.T
    P = T2*(1/float(numDocs))
    P = P.todense() - (alpha0/float(1+alpha0))*np.outer(m,m)
    A = alpha0*(1+alpha0)*P

    # Compute B

    # Get distribution
    k = config['K']
    mu1, alpha1 = fastWhitening(A, M, m, x, k, alpha0, widx)

    print 'Topic probability =', alpha1/alpha0
    # Compute mu
    mu = np.zeros(numWords)
    for i in range(numWords):
        if i != widx:
            mu[i] = mu1[i,0]
    
    p_no_label = np.sum(mu)
    if p_no_label > 1:
        print 'Normalization error !', p_no_label
    else:
        mu[widx] = 1 - p_no_label

    return mu

#---------------------------------------------------------------
# Fast whitening subroutine
#---------------------------------------------------------------   
def fastWhitening(A, M, m, x, k, alpha0, widx):

    A = scipy.sparse.csc_matrix(A)
    print 'SVD 1 ...'
    V, S, Vt = svds(A,k)
    # V is size d x k matrix

    # Form k x k matrix Dhalf, DhalfInv
    Shalf = []
    ShalfInv = []
    for i in range(k):
        s = np.sqrt(S[i])
        Shalf.append(s)
        ShalfInv.append(1/float(s))

    Dhalf = np.diag(Shalf)
    DhalfInv = np.diag(ShalfInv)

    # Compute R = DhalfInv*Vt*B*V*DhalfInv
    W = np.dot(DhalfInv,Vt)
    
    # R is size k x k
    print 'Whitening B'
    R = computeRLDA(W,M,m,widx,alpha0)

    # Compute u the largest singular vector of R
    print 'SVD 2 ...'
    R = scipy.sparse.csc_matrix(R)
    u, s1, ut = svds(R,1)

    # Compute w = V*Dhalf*u
    w = np.dot(V,np.dot(Dhalf,u))
    print 'w row = ', w.shape[0]
    print 'w col = ', w.shape[1]

    # Estimate a = u'*DhalfInv*Vt*x
    xtilde = np.dot(DhalfInv,np.dot(Vt,x))
    a = np.dot(u.transpose(),xtilde)
    
    # Compute mu
    mu = np.dot(w,1/float(a))
    
    return (mu,a**2)

#---------------------------------------------------------------
# Function to compute R matrix (whitened B matrix)
#---------------------------------------------------------------
def computeRLDA(W,M,m,widx,alpha0):

    numWords = M.shape[0]
    print '*number of words =', numWords
    numDocs = M.shape[1]
    print '*number of docs =', numDocs
    K = W.shape[0]
    print '*K =', K
    
    # Compute weights
    print 'Computing matrix R ...'
    M = M.tocsr()
    L = M[widx,:].todense()

    # Compute projected samples
    M = M.tocsc()
    X = np.zeros((K,numDocs),dtype=np.float64)
    X = W*M

    # Y2 = W*M*diag(L) = X*diag[L]
    Y2 = copy.deepcopy(X)
    for i in xrange(numDocs):
        #print L[0,i]
        Y2[:,i] = Y2[:,i]*L[0,i]

    # X1 = W*M*diag(L)*M'*W'/numDocs
    X1 = np.zeros((K,K),dtype=np.float64)
    X1 = np.dot(Y2,X.transpose())
    X1 = X1*(1/float(numDocs))
    print 'X1 done.'

    # X2 = m[l]*W*M*M'*W'/numDocs
    X2 = np.zeros((K,K),dtype=np.float64)
    X2 = np.dot(X,X.transpose()/float(numDocs))
    X2 = X2*m[widx]
    print 'X2 done.'

    # X3 = W*M*diag(L)*Mmat'*W'/numDocs
    X3 = np.zeros((K,K),dtype=np.float64)
    m1 = np.dot(W,m)
    x3 = np.zeros((K,1),dtype=np.float64)
    for k in range(K):
        x3[k] = sum(Y2[k,:])/float(numDocs)

    X3 = np.outer(x3,m1)
    print 'X3 done.'

    # X4 = W*Mmat*diag(L)*M'*W'/numDocs = X3'
    X4 = X3.transpose()
    print 'X4 done.'

    # X5 = m[l]*W*m*m'*W'
    X5 = np.zeros((K,K),dtype=np.float64)
    X5 = m[widx]*np.outer(m1,m1)
    print 'X5 done.'

    # B6 = B1 - (alpha0/(alpha0+2))*(B2+B3+B4) + (2*alpha0^2/((alpha0+2)*(alpha0+1)))*B5
    X6 = np.zeros((K,K),dtype=np.float64)
    X6 = X1 - (alpha0/float(alpha0+2))*(X2+X3+X4) + (2*alpha0**2/float((alpha0+2)*(alpha0+1)))*X5
    print 'X6 done.'

    # B = (alpha0*(alpha0+1)*(alpha0+2)/2)*B6;
    R = (alpha0*(alpha0+1)*(alpha0+2)/float(2))*X6

    return R


if __name__=="__main__":

    if len(sys.argv) <=2:
        print 'Usage: python fastWhitenLDA.py config_file label_word'
        
    else:
        configFile = sys.argv[1]
        labelWord = sys.argv[2]
        #configFile = "configCorpus.txt"
        #configFile = "configCorpus20000.txt"
        config = loadConfig(configFile)
        
        # Preprocessing
        truncMatFile, truncDictFile = preprocess(config)

        # Compute Filter
        splitData(truncMatFile,config)

        # Initialize result directory "res"
        directory = 'res'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Main 
        K = config['K']
        N = config['N']
        
        wordList = [labelWord]
        for word in wordList:

            print '-----------------------------------'
            print 'Labeled word = ' + word
            print '-----------------------------------'

            # Compute topic likelihood vector via whitening method
            tstart = time.time()
            mu = searchByWhiteningLDA(config, word, truncMatFile, truncDictFile)
            print '>> Total runtime = ', time.time() - tstart, ' sec'

            # Print and save top words
            top10List = saveTopWords(truncDictFile, N, mu, word)

            # Compute PMI
            print 'Computing PMI ...'
            pmi = computePMIWhiten(word)
            print 'Topic PMI score =', pmi
        

    

    
