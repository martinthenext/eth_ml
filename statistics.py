#!/usr/bin/env python

import sys
from collections import Counter

from data import Annotation, load_data

annotations = load_data(sys.argv[1])
total = len(annotations)

counter = Counter([annotation.grp for annotation in annotations])
for k, v in counter.items():
  percentage = float(v) / total
  percentage_str = "{0:.0f}%".format(percentage * 100)
  print '|%s|%s|%s|' % (k, v, percentage_str)
print '|Total|%s||' % total
