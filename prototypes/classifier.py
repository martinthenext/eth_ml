#!/usr/bin/env python

'''
ARGUMENT 1: Silver Standard Corpus file
STDOUT: TSV of non-ambiguous annotations
'''

from lxml import etree
import sys
import codecs
import itertools
from collections import Counter

import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)

class Annotation(object):
  def __init__(self, e_element, unit_text):
    self.__slots__ = ['len', 'offset', 'grp', 'text', 'unit_text']

    self.len = e_element.attrib['len']
    self.offset = e_element.attrib['offset']
    self.grp = e_element.attrib['grp']
    self.text = e_element.text.lower()

    self.unit_text = unit_text 

  # for testing ambiguity
  def __hash__(self):
    return hash((self.len, self.offset, self.text))

  def __unicode__(self):
    return '\t'.join(getattr(self, field) for field in self.__slots__)

  GROUP_NAMES = [
    'ACTI', 'ANAT', 'CHEM', 'DEVI',
    'DISO', 'GENE', 'GEOG', 'LIVB', 
    'OCCU', 'ORGA', 'PHYS', 'PROC',
    'PHEN', 'OBJC'
  ]

  GROUP_MAPPING = dict((y, x) for (x, y) in enumerate(GROUP_NAMES))

  def get_group_number(self):
    return self.GROUP_MAPPING[self.grp]

  def get_slices(self):
    begin_highlight = int(self.offset)
    end_highlight = int(self.offset) + int(self.len)

    slice_one = self.unit_text[:begin_highlight]
    slice_two = self.unit_text[begin_highlight:end_highlight]
    slice_three = self.unit_text[end_highlight:]

    return (slice_one, slice_two, slice_three)

  def get_highlighted_repr(self):
    return "%s[[%s]]%s" % self.get_slices()

  def get_context_string(self):
    context_before, _, context_after = self.get_slices()
    return context_before + context_after

# loading XMLs
parser = etree.XMLParser(encoding='utf-8')
ssc = etree.parse(sys.argv[1], parser).getroot()

global_annotations = []

for document in ssc.iter("document"):
  for unit in document.iter("unit"):
    unit_text = unit.find("text").text

    non_empty_e_iter = itertools.ifilter(lambda e: e.text is not None, unit.iter("e"))
    annotations = [Annotation(e, unit_text) for e in non_empty_e_iter]

    # filter out ambiguous annotations
    annotations_counted = Counter(annotations)
    annotations = [annotation for (annotation, count) in 
      filter( lambda (key, count): count == 1, annotations_counted.items())]

    global_annotations += annotations

# classification
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([annotation.get_context_string() for annotation in global_annotations])
y = numpy.array([annotation.get_group_number() for annotation in global_annotations])

classifier = MultinomialNB()
classifier.fit(X, y)

print unicode(global_annotations[0])
print classifier.predict(X[0])
print y[0]