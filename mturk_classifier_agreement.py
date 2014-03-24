#!/usr/bin/env python

''' Train a classifier on non-ambiguous data, classify the mturk data
    and observe the agreement between the classifier and turkers
'''

import sys
import codecs
from data import Annotation
from models import VeryVeryNaiveBayes

#nonambig_annotations = load_data(sys.argv[1])
#classifier = VeryVeryNaiveBayes()
#classifier.train(nonambig_annotations)

# read mturk annotations 
mturk_annotations = []
with codecs.open(sys.argv[1], 'r', 'utf-8') as f:
  for line in f:
    params = dict(zip(['len', 'offset', 'text', 'unit_text', 'grp'], line[:-1].split('|')))
    mturk_annotations.append(Annotation(**params))
