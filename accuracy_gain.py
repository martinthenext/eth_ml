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
  
  return (accuracy_after - accuracy_before)

def get_mean_accuracy_gain(classifier_pickle_file, target_weight, n_runs):
  loaded_classifier = load(classifier_pickle_file)
  loaded_classifier.target_weight = target_weight

  gains = np.zeros(n_runs)

  for i in range(n_runs):
    gains[i] = get_accuracy_gain(loaded_classifier)

  return np.mean(gains)

weights = [10, 50, 100, 500, 1000]
n_runs = 100
mean_gains = Parallel(n_jobs=8)(
  delayed(get_mean_accuracy_gain)('pickles.nobackup/WeightedPartialFitPassiveTransferClassifier2_Medline_fraction0.1', weight, n_runs) 
  for weight in weights
)

print zip(weights, mean_gains)