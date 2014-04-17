#!/usr/bin/env python

''' Train a classifier on non-ambiguous data, classify the mturk data
    and observe the agreement between the classifier and turkers

    ARGUMENT 1: SSC xml file to extract non-ambiguous annotations from
    ARGUMENT 2: CSV file of Mturk majority vote annotations as produced by 
      https://kitt.cl.uzh.ch/kitt/mantracrowd/disambig/vote_results.csv?AgreementThr=0.6
'''

import sys
import codecs
import data 
import models
import numpy as np
from sklearn.externals import joblib
import argparse

def get_agreement(classifier, mturk_labeled_data):
  # read mturk annotations 
  mturk_annotations, labels = mturk_labeled_data

  # classify annotations and output the agreement
  predicted_group_numbers = classifier.predict(mturk_annotations)
  voted_group_numbers = [data.Annotation.GROUP_MAPPING[label] for label in labels]
  agreement = [int(predicted == voted) for predicted, voted in zip(predicted_group_numbers, voted_group_numbers)]

  return np.mean(agreement) 

def get_mturk_classifier_agreement(ssc_file_path, mturk_vote_file_path, classifier_class, **kwargs):
  # train a classifier on unambiguous annotations
  unambig_annotations = data.load_unambiguous_annotations(ssc_file_path)
  classifier = classifier_class(**kwargs)
  classifier.train(unambig_annotations)

  # read mturk annotations 
  mturk_labeled_data = data.load_ambiguous_annotations_labeled(mturk_vote_file_path)

  return get_agreement(classifier, mturk_labeled_data)

def get_mturk_pickled_classifier_agreement(classifier_pickle_file, mturk_vote_file_path, classifier_class, **kwargs):
  classifier = joblib.load(classifier_pickle_file)
  mturk_labeled_data = data.load_ambiguous_annotations_labeled(mturk_vote_file_path)
  return get_agreement(classifier, mturk_labeled_data)

parser = argparse.ArgumentParser()
parser.add_argument("classifier_file_path")
parser.add_argument("vote_csv_file_path")
parser.add_argument("--train", help="First file is a SSC corpus file, train a classifier on it instead of deserializing")

args = parser.parse_args()

if args.train:
  print get_mturk_classifier_agreement(args.classifier_file_path, args.vote_csv_file_path,
   models.OptionAwareNaiveBayesLeftRight)
else:
  print get_mturk_pickled_classifier_agreement(args.classifier_file_path, args.vote_csv_file_path,
   models.OptionAwareNaiveBayesLeftRight)