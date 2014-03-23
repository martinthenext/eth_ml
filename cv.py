#!/usr/bin/env python

import sys

import data
import models

import numpy as np

from sklearn import cross_validation

annotations = data.load_data(sys.argv[1])

def get_error_rate(classifier_class, annotations, nfolds=10):
  sumcostAvg=0
  folds = cross_validation.KFold(len(annotations),n_folds=nfolds)
  classifier = classifier_class()

  annotations = np.array(annotations)

  fold_errors = []
  for train_indices, test_indices in folds:
    classifier.train(annotations[train_indices])
    predictions=classifier.predict(annotations[test_indices])

    errors = [int(annotations[test_index].get_group_number() != prediction) 
     for test_index, prediction in zip(test_indices, predictions)]

    fold_errors.append(np.mean(errors))

  return np.mean(fold_errors)

print get_error_rate(models.VeryVeryNaiveBayes, annotations)