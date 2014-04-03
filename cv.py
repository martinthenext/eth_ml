#!/usr/bin/env python

import sys

import data
import models

import numpy as np

from sklearn import cross_validation

annotations = data.load_unambiguous_annotations(sys.argv[1])

class CountPrinter:
  def __init__(self, total):
    self.total = total
    self.current = 0

  def count(self):
    print '%s/%s' % (self.current, self.total)
    self.current += 1

def get_error_rate(classifier_class, annotations, n_folds=10, verbose=False, **kwargs):
  folds = cross_validation.KFold(len(annotations), n_folds=n_folds)
  classifier = classifier_class(**kwargs)

  annotations = np.array(annotations)

  counter = CountPrinter(n_folds)
  fold_errors = []
  for train_indices, test_indices in folds:
    if verbose: counter.count()

    classifier.train(annotations[train_indices])
    predictions = classifier.predict(annotations[test_indices])

    errors = [int(annotations[test_index].get_group_number() != prediction) 
     for test_index, prediction in zip(test_indices, predictions)]

    fold_errors.append(np.mean(errors))

  return np.mean(fold_errors)

print get_error_rate(models.NaiveBayesContextRestricted, annotations, 10, True, window_size=3)
