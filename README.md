# Fast Topic Search

A python implementation of a fast practical topic search algorithm

The algorithm called Topic Search by Whitening is described in the following paper:

Avik Ray, Joe Neeman, Sujay Sanghavi and Sanjay Shakkottai, “The Search Problem in Mixture Models”, Journal of Machine Learning Research (JMLR), vol. 18, no. 206, pp. 1-61, April 2018. (http://www.jmlr.org/papers/volume18/16-483/16-483.pdf)

Please cite the above paper if you are using the code.

Copyright (C) 2016 Avik Ray

Copyright of scripts uci_to_scipy.py and truncate_vocabulary.py belong
to Yoni Halpern et al. (authors of the paper "A Practical Algorithm for 
Topic Modeling with Provable Guarantees", ICML 2013)

Contact: avik@utexas.edu

## Files and Folder description:

fastWhitenLDA.py -- Main implementation of the Topic Search by Whitening 
                    algorithm

utils.py -- Utility functions used by the topic search algorithm

uci_to_scipy.py -- Converts UCI format corpus data (bag-of-words) to
		   sparse scipy matrix

truncate_vocabulary -- Used to remove rare words, stopwords, and 
                       truncate the input vocabulary


Folder nips_dataset -- Contains the NIPS bag-of-words dataset from UCI
                       repository, corresponding list of stopwords and 
                       demo config file for topic search algorithm.

Folder nytimes_dataset -- Contains list of stopwords and demo config 
                     file for NY Times bag-of-words dataset. The 
                     dataset can be downloaded from UCI repository at
                     http://archive.ics.uci.edu/ml/datasets/Bag+of+Words


## Usage: Performing topic search via label word

### Files required in same directory:

config_file -- Test settings for topic search algorithm
               (config parameter description below)

corpus_file -- UCI format bag-of-words corpus file

vocab_file -- Word vocabulary for the corpus

stopwords_file -- List of stopwords

### Environment:

Python 2.x with numpy, scipy 

### Usage:

1) Copy the above four files into the main directory 
   (e.g. for NIPS dataset all files in nips_dataset folder to main folder)

2) Run from shell/command prompt

```
python fastWhitenLDA.py config_file label_word
```

where "label_word" is a single word whose corresponding topic is 
being searched in the document corpus.

### Output:

The top N words (N specified in config file) in the target topic is
saved in a text file under directory called "res" and also displayed
by the program. 

The topic PMI score is also computed and displayed. However this 
requires internet access to Palmetto web service to compute topic PMI 
(see https://github.com/AKSW/Palmetto/wiki/How-Palmetto-can-be-used).
Note that if this service is down the program may output error messages
and display PMI score = 0. In such cases the PMI score maybe computed
offline by the Palmetto program (see instructions in the above link).

(Note that the program will also generate several intermediate files
like truncated matrix, truncated vocabulary, filter file etc.) 

## Config file parameter description:

corpus_file -- Name of corpus file (UCI bag-of-words format)

dictionary_file -- Name of vocabulary file

mat_file_name -- Name of scipy sparse matrix file 

filter_file -- Name of test, train, validation split filter file

stopwords_file -- Name of stopwords file

train_fraction -- Fraction of documents used for training

validation_fraction -- Fraction of documents used for validation

test_fraction -- Fraction of documents used for test

rare_word_threshold -- Vocabulary truncation parameter. Words present in
                    less than rare_word_threshold documents are removed

num_topics -- Total number of topics in corpus. This is the parameter k 
              in topic search algorithm

top_words_to_display -- Number of top words from the topic to be displayed
                        and saved in output file

alpha_0 -- Dirichlet parameter alpha_0 in topic search algorithm

weight -- Weight parameter to weigh labeled documents (currently not used)


END_OF_README
