# -*- coding: utf-8 -*-

from __future__ import absolute_import, division
import os
import unicodedata
from os import listdir
from os.path import isfile, join
import datetime
import urllib2
import yaml
from urllib import urlencode, quote_plus, quote
import urlparse
from web2dehc.fetcher import Fetcher
import yakonfig
import glob
import dblogger
from collections import defaultdict, Counter
import argparse
import logging
import numpy as np

import streamcorpus
import streamcorpus_pipeline
from streamcorpus import Chunk as SC_Chunk
from streamcorpus_pipeline._pipeline import PipelineFactory
from streamcorpus_pipeline.stages import PipelineStages

from dossier.fc import FeatureCollectionChunk as FC_Chunk
from dossier.fc import StringCounter
from dossier.models.etl.interface import html_to_fc

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import LeavePLabelOut

logger = logging.getLogger(__name__)
'''
Use web2dehc to create the corpus 
web2dehc noncyber.txt corpus/noncyber --skip-push-dehc -k GOOGLE_API_KEY
web2dehc cyber.txt corpus/cyber --skip-push-dehc -k GOOGLE_API_KEY    
    cyber_urls = [
        # English
        'http://thehackernews.com/',
        'https://www.blackhat.com/',
        'http://www.securityweek.com/',
        
        # Russian
        'https://xakep.ru/',
        'http://www.gofuckbiz.com/',
        'http://antichat.ru/',
        'https://forum.xeksec.com/',
        
        # Chinese
        'http://forum.cnsec.org/',
        'http://forum.eviloctal.com/',
        'http://bbs.pediy.com/',
        'http://bbs.blackbap.org/forum.php',
    ]

    noncyber_urls = [
        # English
        'https://www.yahoo.com/',
        'http://techcrunch.com/',
        'http://www.engadget.com/',
        'http://www.nytimes.com/',
        'http://www.cnn.com/',
        
        # Russian
        'http://www.pravda.ru/',
        'http://www.gazeta.ru/',
        'http://lifenews.ru/',
        'http://www.bbc.co.uk/russian/science',

        # Chinese
        'http://www.163.com/',
        'http://www.56.com/',
        'http://tech.sina.com.cn/',
        'http://cn.engadget.com/'
    ]
'''

def run_scp(yaml_path, chunk_path, corpus_name):
    '''Run the streamcorpus pipeline using the YAML reader
    
    `yaml_path` is the path to the yaml file generated from the
    generate_yaml function

    `chunk_path` is the directory the chunk file will be saved to

    `corpus_name` is the name of the saved chunk file

    '''
    
    config = yakonfig.get_global_config()
    scp_config = config['streamcorpus_pipeline']
    stages = PipelineStages()
    factory = PipelineFactory(stages)
    scp_config['to_local_chunks']['output_path'] = chunk_path
    scp_config['to_local_chunks']['output_name'] = corpus_name
    pipeline = factory(scp_config)

    # pass in the yaml file
    pipeline.run(yaml_path)

def feature_pipeline(chunk_in, FC_chunk_out):
    '''Run a basic pipeline to generate feature collections from
    streamitems. If file exists just loads the existing file. Returns
    a list of either the generated FCs or the FCs in the existing
    file.
    
    `chunk_in` path to SC chunk file

    `FC_chunk_out` path where the FC chunk file be written

    '''    
    if isfile(FC_chunk_out):
        print FC_chunk_out, 'already exists...',
        fcs = [fc for fc in FC_Chunk(FC_chunk_out, mode='rb')]
        print 'loaded.'
    else:
        chunk_out = FC_Chunk(FC_chunk_out, mode='wb')
        fcs = []
        for cfile in glob.glob(join(chunk_in,'*.sc.xz')):
            print 'processing', cfile
            for i, si in enumerate(SC_Chunk(cfile)):
                if i % 10==0: print i, 'fc processed'
                fc = html_to_fc(
                    html=si.body.raw,
                    encoding=si.body.encoding,
                    url=si.abs_url)
                chunk_out.add(fc)
                fcs.append(fc)

        print 'done creating', FC_chunk_out
    return fcs
    
def build_fcs(path, cyber_label, noncyber_label):
    '''Builds FCs for the cyber and noncyber data
    
    `path` path to the corpus

    `cyber_label` path to where the cyber file will be

    `noncyber_label` path to where the noncyber files will be

    '''        
    cyber_path = join(path, cyber_label)
    cyber_out = join(cyber_path, 'cyber.fc')
    noncyber_path = join(path, noncyber_label)
    noncyber_out = join(noncyber_path, 'noncyber.fc')

    cyber_fcs = feature_pipeline(cyber_path, cyber_out)
    noncyber_fcs = feature_pipeline(noncyber_path, noncyber_out)

    return cyber_fcs, noncyber_fcs

class Clacy(object):
    def __init__(self, clf=None):
        '''Class for doing cyber classification.

        `path` is an sklearn classifier. If not set it defaults to
        Naive Bayes.

        '''
        
        self.v = DictVectorizer(sparse=False)
        if clf == None:
            self.clf = BernoulliNB()
        else:
            self.clf = clf

    def load_corpus(self, cyber_fcs, noncyber_fcs, features):
        '''Vectorize the corpus and into features that can be used by sklearn

        `cyber_fcs` iterator of cyber fcs

        `noncyber_fcs` iterator of noncyber fcs

        `features` which features to use in the fcs for classification
        '''
        
        self.labels = np.array([1] * len(cyber_fcs) + [0] * len(noncyber_fcs))
        self.features = features
        D = list()

        # Url map store which fc came from which url which is
        # important for cross-validation over websites to check
        # generalization
        url_map = list()
        urls = list()
        for fc in (cyber_fcs + noncyber_fcs):
            feat = StringCounter()

            for f in features:
                feat += fc[f]

            netloc = urlparse.urlparse(fc['meta_url']).netloc
            if netloc not in url_map:
                url_map.append(netloc)

            urls.append(url_map.index(netloc))
            D.append(feat)

        self.urls = np.array(urls)
        self.X = self.v.fit_transform(D)

    def cv(self):
         '''Evaluate the classifier by running cross-validation by splitting on
         the url basename to test cross-site generalization
         
         returns a list of F1 scores for each CV split. 
         '''
        lplo = LeavePLabelOut(self.urls, p=1)
        return cross_validation.cross_val_score(self.clf, self.X, self.labels,
                                                cv=lplo, scoring='f1')

    def train(self):
        '''Trains the classifier on the data which was loaded through self.load_corpus        
        '''
        self.clf.fit(self.X, self.labels)

    def predict(self, fcs):
        '''Predict cyber or noncyber for a new set of fcs. 

        `fcs` new fcs to predict the label for

        returns a list of 0s and 1s where 0s correspond to noncyber and 1s correspond to cyber. 
        '''
        
        D = []
        for fc in (fcs):
            feat = StringCounter()

            for f in self.features:
                feat += fc[f]

            D.append(feat)

        X = self.v.transform(D)        
        return self.clf.predict(X)
        
if __name__ == '__main__':
    path = 'corpus'
    cyber_label = 'cyber'
    noncyber_label = 'noncyber'

    # build_fcs will take all the .sc.xz files in path/cyber_label and
    # path/noncyber_label as the labeled corpus.
    cyber_fcs, noncyber_fcs = build_fcs(path, cyber_label, noncyber_label)

    # Initialize the classifier with a custom sklean classifier
    c = Clacy(svm.SVC(kernel='linear', C=1))
    # Load in the data
    c.load_corpus(cyber_fcs, noncyber_fcs, ['bowNP'])
    # Cross-validate on the data
    scores = c.cv()
    print np.mean(scores)
    print scores
    # Train on the full corpus
    c.train()
    # Make some predictions
    predictions = c.predict([cyber_fcs[0], noncyber_fcs[0]])
    print predictions

