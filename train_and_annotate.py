#!/usr/bin/env python

import argparse
import data
import models
from sklearn.externals import joblib

CLASSIFIER_PICKE_DIR = 'pickles.nobackup/'

def get_classifier_serialization_filename(classifier_name, ssc_file_list):
  corpus_codes = [sss_file_name.split('/')[-1].split("-")[0] for sss_file_name in ssc_file_list]
  return '%s_trained_on_%s' % (classifier_name, '_'.join(corpus_codes))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train', nargs='+', type=str)
  parser.add_argument('-a', '--annotate', type=str)
  args = parser.parse_args()

  annotations = []
  for ssc_file in args.train:
    annotations += data.load_unambiguous_annotations(ssc_file)

  classifier = models.OptionAwareNaiveBayesLeftRightCutoff(window_size = 5, cutoff = 9)
  classifier.train(annotations)

  classifier_filename = get_classifier_serialization_filename('NaiveBayesWsCo9', args.train)
  joblib.dump(classifier, serialization_path)

