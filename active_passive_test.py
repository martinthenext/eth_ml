#!/usr/bin/env python

'''

This file unpickles your transfer learning classifier, feeds training examples to it
and measures the progression of accuracy

See magic constants in the bottom

'''

from mturk_classifier_agreement import get_agreement
from data import load_ambiguous_annotations_labeled
from sklearn.cross_validation import train_test_split
import numpy as np
import itertools
from copy import deepcopy
# TODO make the deprecation warning go away
from sklearn.externals import joblib
import random
import plot_curves
from cv import CountPrinter

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

    self.annotations = np.array(annotations)
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

class UncertaintySamplingLeastConfidenceActiveLearner(PassiveLearner):
  def pop_index_from_pool(self):
    pool_indexes = list(self.index_pool)
    pool_confidences = self.classifier.get_prob_estimates(self.annotations[pool_indexes])
    # pick the index of the least confident prediction
    min_confidence_pool_index = np.argmin(pool_confidences)
    index = pool_indexes[min_confidence_pool_index]
    self.index_pool.remove(index)
    return index


def get_accuracy_progression(train_test_set, classifier_to_measure, annotations, labels, target_weight, learner_class):
  pool_annotations, test_annotations, pool_labels, test_labels = train_test_set
  accuracy_progression = np.zeros(len(pool_annotations) + 1)

  learner = learner_class(classifier_to_measure, pool_annotations, pool_labels, target_weight = target_weight)

  # initialize the accuracy list with the initial accuracy
  accuracy_progression[0] = get_agreement(learner.classifier, (test_annotations, test_labels))

  for i in range(1, len(accuracy_progression)):
    learner.learn()
    accuracy_progression[i] = get_agreement(learner.classifier, (test_annotations, test_labels))

  return accuracy_progression

PLOT_FOLDER = 'plots/'
CLASSIFIER_PICKLE_FOLDER = 'pickles.nobackup/'
ANNOTATIONS_LABELED_FILENAME = '../vote_results_thr0.75-new6.csv'

def plot_learning_curves(classifier_pickle_filename, target_weight=1000, n_simulations=100, test_size=0.33):

  classifier_loaded = joblib.load(CLASSIFIER_PICKLE_FOLDER + classifier_pickle_filename)
  annotations_loaded, labels_loaded = load_ambiguous_annotations_labeled(ANNOTATIONS_LABELED_FILENAME)

  pool, _, _, _ = train_test_split(annotations_loaded, labels_loaded, test_size = test_size) 
  n_iterations = len(pool) + 1

  passive_accuracy = np.zeros((n_simulations, n_iterations))
  active_accuracy = np.zeros((n_simulations, n_iterations))

  counter = CountPrinter(n_simulations)

  for run_number in range(n_simulations):
    # securing statelessness
    classifier = deepcopy(classifier_loaded)
    annotations= deepcopy(annotations_loaded)
    labels = deepcopy(labels_loaded)

    train_test_set = train_test_split(annotations, labels, test_size = test_size) 

    passive_accuracy[run_number] = get_accuracy_progression(
      train_test_set, classifier, annotations, labels, target_weight, PassiveLearner)
    active_accuracy[run_number] = get_accuracy_progression(
      train_test_set, classifier, annotations, labels, target_weight, UncertaintySamplingLeastConfidenceActiveLearner)

    counter.count()

  passive_avg_accuracy_progression = np.mean(passive_accuracy, axis=0)
  active_avg_accuracy_progression = np.mean(active_accuracy, axis=0)

  plot_filename = PLOT_FOLDER + classifier_pickle_filename + '_weight' + str(target_weight)

  plot_curves.plot_curves(plot_filename, title="Average iteration accuracy for %s simulations" % n_simulations,
   PassiveLearner=passive_avg_accuracy_progression, ActiveLearner=active_avg_accuracy_progression)


classifiers = ['WeightedSVMPartialFitPassiveTransferClassifier_Medline', 'WeightedSVMHuberPartialFitPassiveTransferClassifier_Medline']
weights = [10, 100, 1000]
combinations = list(itertools.product(classifiers, weights))

joblib.Parallel(n_jobs=len(combinations))(joblib.delayed(plot_learning_curves)(classifier, weight) for classifier, weight in combinations)
