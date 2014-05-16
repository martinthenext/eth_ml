#!/usr/bin/env python

'''

This will train a transfer classifier on unambiguous data and serialize it to file with sklearn.joblib

ARGUMENT 1: SSC xml file to extract non-ambiguous annotations from

Load like: joblib.load('pickles.nobackup/EMEA_OptionAwareNaiveBayesLeftRight')

'''

import transfer
import data
import sys
from sklearn.externals import joblib
import itertools

def train_and_serialize(unambig_annotations, serialization_path, classifier_class, transfer=True, dataset_fraction=None, **kwargs):
  if dataset_fraction:
    step = int(1/dataset_fraction)
    train_set = itertools.islice(unambig_annotations, 0, None, step)
    serialization_path += '_fraction' + str(dataset_fraction)
  else:
    train_set = unambig_annotations
  classifier = classifier_class(**kwargs)
  if transfer:
    classifier.train_source(train_set)
  else:
    classifier.train(train_set)

  joblib.dump(classifier, serialization_path)

print 'loading annotations'
unambig_annotations = data.load_unambiguous_annotations(sys.argv[1])
print 'finished loading'

def run(frac):
  train_and_serialize(unambig_annotations, 'pickles.nobackup/WeightedPartialFitPassiveTransferClassifier2_Medline', 
    transfer.WeightedPartialFitPassiveTransferClassifier2, dataset_fraction=frac)
  return None

joblib.Parallel(n_jobs=4)(joblib.delayed(run)(frac) for frac in [None, 0.1, 0.05, 0.01])
