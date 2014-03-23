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

  for train_indices, test_indices in folds:
    trainset=np.array([annotations[i] for i in train_indices[:]])
    classifier.train(trainset)
    testset=np.array([annotations[i] for i in test_indices[:]])
    predictions=classifier.predict(testset)
    sumcost=0
    for i in range(len(predictions)):
      sumcost+=int(predictions[i] != annotations[test_indices[i]].get_group_number())
    sumcostAvg+=float(sumcost)/float(len(testset))

  return float(sumcostAvg)/float(nfolds)

print get_error_rate(models.VeryVeryNaiveBayes, annotations)