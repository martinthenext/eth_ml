#!/usr/bin/env python

import sys

import data
import models

annotations = data.load_data(sys.argv[1])

def get_error_rate(classifier_class, annotations):
  classifier = classifier_class()
  classifier.train(annotations[:-1])
  return int(classifier.predict([annotations[-1]]) == annotations[-1].get_group_number())

print get_error_rate(models.VeryVeryNaiveBayes, annotations)