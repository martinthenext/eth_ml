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

def get_mturk_classifier_agreement(ssc_file_path, mturk_vote_file_path, classifier_class, **kwargs):
  # train a classifier on unambiguous annotations
  unambig_annotations = data.load_unambiguous_annotations(ssc_file_path)
  classifier = classifier_class(**kwargs)
  classifier.train(unambig_annotations)

  # read mturk annotations 
  mturk_annotations, labels = data.load_ambiguous_annotations_labeled(mturk_vote_file_path)

  # classify annotations and output the agreement
  predicted_group_numbers = classifier.predict(mturk_annotations)
  voted_group_numbers = [data.Annotation.GROUP_MAPPING[label] for label in labels]
  agreement = [int(predicted == voted) for predicted, voted in zip(predicted_group_numbers, voted_group_numbers)]

  return np.mean(agreement)

print get_mturk_classifier_agreement(sys.argv[1], sys.argv[2], models.OptionAwareRandom)
