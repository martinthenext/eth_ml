#!/usr/bin/env python

'''

This will train a transfer classifier on unambiguous data and serialize it to file with sklearn.joblib

ARGUMENT 1: SSC xml file to extract non-ambiguous annotations from
ARGUMENT 2: file name to serialize the classifier to

Load like: joblib.load('pickles.nobackup/EMEA_OptionAwareNaiveBayesLeftRight')

'''

import transfer
import data
import sys
from sklearn.externals import joblib

def train_and_serialize(ssc_file_path, serialization_path, classifier_class, **kwargs):
  unambig_annotations = data.load_unambiguous_annotations(ssc_file_path)
  classifier = classifier_class(**kwargs)
  classifier.train_source(unambig_annotations)

  joblib.dump(classifier, serialization_path)

train_and_serialize(sys.argv[1], sys.argv[2], transfer.WeightedPartialFitPassiveTransferClassifier, target_weight = 10)
