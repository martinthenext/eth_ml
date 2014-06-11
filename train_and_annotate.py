#!/usr/bin/env python

import argparse
import data
import models
from sklearn.externals import joblib
from lxml import etree
import itertools

CLASSIFIER_PICKE_DIR = 'pickles.nobackup/'

def get_classifier_serialization_filename(classifier_name, ssc_file_list):
  corpus_codes = [sss_file_name.split('/')[-1].split("-")[0] for sss_file_name in ssc_file_list]
  return '%s_trained_on_%s' % (classifier_name, '_'.join(corpus_codes))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train', nargs='+', type=str)
  parser.add_argument('-a', '--annotate', type=str)
  args = parser.parse_args()

  # Check if there is an appropriate pickle already just load it instead of training
  classifier_filename = get_classifier_serialization_filename('NaiveBayesWsCo9', args.train)
  classifier_path = CLASSIFIER_PICKE_DIR + classifier_filename
  try:
    classifier = joblib.load(classifier_path)
    print 'classifier loaded from pickle'
  except IOError:
    # Load annotations
    annotations = []
    for ssc_file in args.train:
      annotations += data.load_unambiguous_annotations(ssc_file)    
    # Train the classifier
    classifier = models.OptionAwareNaiveBayesLeftRightCutoff(window_size = 5, cutoff = 9)
    classifier.train(annotations)
    # Picke the classifier
    joblib.dump(classifier, CLASSIFIER_PICKE_DIR + classifier_filename)

  # SSC XML
  parser = etree.XMLParser(encoding='utf-8')
  ssc = etree.parse(args.annotate, parser).getroot()

  for document in ssc.iter("document"):
    for unit in document.iter("unit"):
      unit_text = unit.find("text").text

      non_empty_e = itertools.ifilter(lambda e: e.text is not None, unit.iter("e"))
      annotations = [data.Annotation(e, unit_text) for e in non_empty_e]
      
      
