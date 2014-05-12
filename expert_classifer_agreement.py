import sys
import codecs
import models
import numpy as np
from sklearn.externals import joblib
import argparse
import data

def get_expert_agreement(classifier, mturk_labeled_data):
  # read mturk annotations
  mturk_annotations, labels = mturk_labeled_data
  # classify annotations and output the agreement
  predicted_group_numbers = classifier.predict(mturk_annotations)
  voted_group_numbers, indices = zip(*[ (data.Annotation.GROUP_MAPPING[labels[index]], index) for index in range(len(labels)) 
                          if labels[index] in data.Annotation.GROUP_NAMES ])

  voted_group_numbers_strict = [-1] * len(predicted_group_numbers)
  voted_group_numbers_strict = np.array(voted_group_numbers_strict)
  voted_group_numbers_strict[list(indices)] = voted_group_numbers
  restricted_group_numbers = predicted_group_numbers[list(indices)]
  useful_agreement = [int(predicted == voted) for predicted, voted in zip(restricted_group_numbers, voted_group_numbers)]
  strict_agreement = [int(predicted == voted) for predicted, voted in zip(predicted_group_numbers, voted_group_numbers_strict)]

  return (np.mean(strict_agreement),np.mean(useful_agreement) )

def get_mturk_pickled_classifier_agreement(classifier_pickle_file, mturk_vote_file_path, **kwargs):
  classifier = joblib.load(classifier_pickle_file)
  mturk_labeled_data = data.load_ambiguous_annotations_labeled_generic(mturk_vote_file_path)
  return get_expert_agreement(classifier, mturk_labeled_data)

a,b = get_mturk_pickled_classifier_agreement(sys.argv[1],sys.argv[2])
print a
print b