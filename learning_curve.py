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
import itertools
from copy import deepcopy
# TODO make the deprecation warning go away
from sklearn.externals import joblib

classifier_pickle_filename = sys.argv[1]
annotations_labeled_filename = sys.argv[2]

def get_classifier_agreement_increase_table(target_weight_list, n_simulations = 1000):
  agreement_before = np.zeros(n_simulations)
  agreement_after = np.zeros(n_simulations)
  annotations, labels = load_ambiguous_annotations_labeled(annotations_labeled_filename)
  result = ""

  for weight in target_weight_list:
    for i in xrange(n_simulations):
      classifier = joblib.load(classifier_pickle_filename)

      pool_annotations, test_annotations, pool_labels, test_labels = train_test_split(
        annotations, labels, test_size = 0.33)  

      # validate the initial state of the classifier
      agreement_before[i] = get_agreement(classifier, (test_annotations, test_labels))

      # test: target train on the entire pool, validate again 
      classifier.target_weight = weight
      classifier.train_target_online(pool_annotations, pool_labels)
      agreement_after[i] = get_agreement(classifier, (test_annotations, test_labels))

    result += str(weight), np.mean(agreement_after - agreement_before)
  return result

# this does not change the state of classifier
def get_accuracy_progression(classifier_to_measure, annotations, labels, target_weight):
  classifier = deepcopy(classifier_to_measure)
  classifier.target_weight = target_weight

  pool_annotations, test_annotations, pool_labels, test_labels = train_test_split(
        annotations, labels, test_size = 0.33) 

  # initialize the accuracy list with the initial accuracy
  accuracy_list = [ get_agreement(classifier, (test_annotations, test_labels)) ]
  for pool_annotation, pool_label in itertools.izip(pool_annotations, pool_labels):
    classifier.train_target_online([pool_annotation], [pool_label])
    accuracy_list.append( get_agreement(classifier, (test_annotations, test_labels)) )

  return accuracy_list

def diff_iter(seq):
  return (y - x for x, y in
   itertools.izip(itertools.islice(seq, 0, len(seq) - 1), itertools.islice(seq, 1, len(seq)))
  )

def format_flot_list(seq, sep=" "):
  result = ""
  for item in seq:
    result += "%.2f" % item
    result += sep
  return result

classifier = joblib.load(classifier_pickle_filename)
annotations, labels = load_ambiguous_annotations_labeled(annotations_labeled_filename)

accuracy_progression = get_accuracy_progression(classifier, annotations, labels, 1000)
print format_flot_list(accuracy_progression)
print format_flot_list(list(diff_iter(accuracy_progression)))