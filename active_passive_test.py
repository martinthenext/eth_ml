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
import random

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

''' Wrap a classifier into this to train passively
'''
class PassiveLearner(object):
  def __init__(self, classifier, annotations, labels, **kwargs):
    self.classifier = deepcopy(classifier)
    for key, value in kwargs.items():
      setattr(self.classifier, key, value)

    self.annotations = annotations
    self.labels = labels
    self.index_pool = set(range(len(annotations)))

  def pop_index_from_pool(self):
    training_index = random.sample(self.index_pool, 1)[0]
    self.index_pool.remove(training_index)
    return training_index

  def learn(self):
    if self.index_pool:
      index = self.pop_index_from_pool()
      annotation, label = self.annotations[index], self.labels[index]
      self.classifier.train_target_online([annotation], [label])
      print 'trained on point %s' % index

class UncertaintySamplingLeastConfidenceActiveLearner(PassiveLearner):
  def pop_index_from_pool(self):
    confidence = self.classifier.get_prob_estimates(self.annotations)
    # pick the index of the least confident prediction
    return np.argmin(confidence)


def get_accuracy_progression(train_test_set, classifier_to_measure, annotations, labels, target_weight, learner_class):
  pool_annotations, test_annotations, pool_labels, test_labels = train_test_set

  learner = learner_class(classifier_to_measure, pool_annotations, pool_labels, target_weight = 1000)

  # initialize the accuracy list with the initial accuracy
  accuracy_list = [ get_agreement(learner.classifier, (test_annotations, test_labels)) ]

  for _ in pool_annotations:
    learner.learn()
    accuracy_list.append( get_agreement(learner.classifier, (test_annotations, test_labels)) )

  return accuracy_list


def diff_iter(seq):
  return (y - x for x, y in
   itertools.izip(itertools.islice(seq, 0, len(seq) - 1), itertools.islice(seq, 1, len(seq)))
  )

def format_float_list(seq, sep=" "):
  result = ""
  for item in seq:
    result += "%.2f" % item
    result += sep
  return result

classifier = joblib.load(classifier_pickle_filename)

annotations, labels = load_ambiguous_annotations_labeled(annotations_labeled_filename)
train_test_set = train_test_split(annotations, labels, test_size = 0.33) 

accuracy_progression = get_accuracy_progression(train_test_set, classifier, annotations, labels, 1000, PassiveLearner)
print 'Passive Learner'
print format_float_list(accuracy_progression)
print format_float_list(list(diff_iter(accuracy_progression)))

classifier = joblib.load(classifier_pickle_filename)

accuracy_progression = get_accuracy_progression(train_test_set, classifier, annotations, labels, 1000, UncertaintySamplingLeastConfidenceActiveLearner)
print 'Active Learner'
print format_float_list(accuracy_progression)
print format_float_list(list(diff_iter(accuracy_progression)))