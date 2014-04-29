#!/usr/bin/env python

'''

This file unpickles your transfer learning classifier, feeds training examples to it
and measures the progression of accuracy

ARGUMENT 1: Pickle of the transfer learning classifier
ARGUMENT 2: CSV file of Mturk majority vote annotations as produced by 
      https://kitt.cl.uzh.ch/kitt/mantracrowd/disambig/vote_results.csv?AgreementThr=0.6

'''

from mturk_classifier_agreement import get_agreement
from data import load_ambiguous_annotations_labeled
from sklearn.cross_validation import train_test_split
import numpy as np
import sys
# TODO make the deprecation warning go away
from sklearn.externals import joblib

N_SIMULATIONS = 100
agreement_before = np.zeros(N_SIMULATIONS)
agreement_after = np.zeros(N_SIMULATIONS)

annotations, labels = load_ambiguous_annotations_labeled(sys.argv[2])

for i in xrange(N_SIMULATIONS):
  classifier = joblib.load(sys.argv[1])

  pool_annotations, test_annotations, pool_labels, test_labels = train_test_split(
    annotations, labels, test_size = 0.33)  

  # validate the initial state of the classifier
  agreement_before[i] = get_agreement(classifier, (test_annotations, test_labels))

  # test: target train on the entire pool, validate again 
  classifier.target_weight = 10000
  classifier.train_target_online(pool_annotations, pool_labels)
  agreement_after[i] = get_agreement(classifier, (test_annotations, test_labels))

print np.mean(agreement_after - agreement_before)