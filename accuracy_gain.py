#!/usr/bin/env python
from data import load_ambiguous_annotations_labeled
from mturk_classifier_agreement import get_agreement
from sklearn.externals.joblib import load, Parallel, delayed
from train_and_serialize import train_and_serialize
from copy import deepcopy
from sklearn.cross_validation import train_test_split
import numpy as np

MTURK_VOTE_FILE = '../vote_results_thr0.75-new6.csv'
annotations, labels = load_ambiguous_annotations_labeled(MTURK_VOTE_FILE)

''' This function gets an accuracy gain for RANDOM train/test split of data
''' 
def get_accuracy_gain(loaded_classifier):
  classifier = deepcopy(loaded_classifier)
  pool_annotations, test_annotations, pool_labels, test_labels = train_test_split(
        annotations, labels, test_size = 0.33)

  accuracy_before = get_agreement(classifier, (test_annotations, test_labels))
  classifier.train_target_online(pool_annotations, pool_labels)
  accuracy_after = get_agreement(classifier, (test_annotations, test_labels))
  
  return (accuracy_after, accuracy_before)

def get_mean_accuracy_gain(classifier_pickle_file, target_weight, n_runs):
  loaded_classifier = load(classifier_pickle_file)
  loaded_classifier.target_weight = target_weight

  accuracies_before = np.zeros(n_runs)
  accuracies_after = np.zeros(n_runs)
  gains = np.zeros(n_runs)

  for i in range(n_runs):
    accuracy_after, accuracy_before = get_accuracy_gain(loaded_classifier)
    accuracies_before[i] = accuracy_before
    accuracies_after[i] = accuracy_after
    gains[i] = accuracy_after - accuracy_before

  return np.mean(accuracies_before), np.mean(accuracies_after), np.mean(gains)

weights = [10, 100, 500, 1000]
n_runs = 100
results = Parallel(n_jobs=4)(
  delayed(get_mean_accuracy_gain)('pickles.nobackup/WeightedSVMHuberPartialFitPassiveTransferClassifier_Medline', weight, n_runs) 
  for weight in weights
)

for weight, result in zip(weights, results):
  print '%s\t%s\t%s\t%s' % (weight, result[0], result[1], result[2])