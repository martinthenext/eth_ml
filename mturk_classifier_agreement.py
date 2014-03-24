#!/usr/bin/env python

''' Train a classifier on non-ambiguous data, classify the mturk data
    and observe the agreement between the classifier and turkers

    ARGUMENT 1: SSC xml file to extract non-ambiguous annotations from
    ARGUMENT 2: CSV file of Mturk majority vote annotations
'''

import sys
import codecs
from data import load_data, Annotation
from models import VeryVeryNaiveBayes
import numpy as np

# train a classifier on non-ambiguous annotations

nonambig_annotations = load_data(sys.argv[1])
classifier = VeryVeryNaiveBayes()
classifier.train(nonambig_annotations)

# read mturk annotations 
mturk_annotations = []
with codecs.open(sys.argv[2], 'r', 'utf-8') as f:
  for line in f:
    params = dict(zip(['len', 'offset', 'text', 'unit_text', 'grp'], line[:-1].split('|')))
    mturk_annotations.append(Annotation(**params))

# classify annotations and output the agreement
predicted_group_numbers = classifier.predict(mturk_annotations)
voted_group_numbers = [annotation.get_group_number() for annotation in mturk_annotations]
agreement = [int(predicted == voted) for predicted, voted in zip(predicted_group_numbers, voted_group_numbers)]
print np.mean(agreement)
