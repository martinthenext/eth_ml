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
import itertools

def train_and_serialize(ssc_file_path, serialization_path, classifier_class, transfer=True, dataset_fraction=None, **kwargs):
  unambig_annotations = data.load_unambiguous_annotations(ssc_file_path)
  if dataset_fraction:
    step = int(1/dataset_fraction)
    train_set = itertools.islice(unambig_annotations, 0, None, step)
  else:
    train_set = unambig_annotations
  classifier = classifier_class(**kwargs)
  if transfer:
    classifier.train_source(unambig_annotations)
  else:
    classifier.train(unambig_annotations)

  joblib.dump(classifier, serialization_path)

train_and_serialize(sys.argv[1], 'pickles.nobackup/WeightedPartialFitPassiveTransferClassifier_Medline_weight1000_fraction025', 
  transfer.WeightedPartialFitPassiveTransferClassifier, dataset_fraction=0.25, target_weight=1000)