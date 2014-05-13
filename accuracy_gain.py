#!/usr/bin/env python
from data import load_ambiguous_annotations_labeled
from mturk_classifier_agreement import get_agreement
from sklearn.externals import joblib
from train_and_serialize import train_and_serialize

if __name__ == "__main__":
  MTURK_VOTE_FILE = '../vote_results_thr0.75-new5.csv'
  #CLASSIFIER_PICKLE_FILE = 'pickles.nobackup/WeightedPartialFitPassiveTransferClassifier_Medline_2'
  CLASSIFIER_PICKLE_FILE = 'pickles.nobackup/Medline_OptionAwareNaiveBayes_window20'
  
  annotations, labels = load_ambiguous_annotations_labeled(MTURK_VOTE_FILE)
  classifier = joblib.load(CLASSIFIER_PICKLE_FILE)

  print get_agreement(classifier, (annotations, labels))  