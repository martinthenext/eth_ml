''' Load data from the SSC xml file 
'''

from lxml import etree
import sys
import codecs
import itertools
from collections import Counter

''' Annotation container class. Supply kwargs to initialize class fields by name
'''
class Annotation(object):
  def __init__(self, e_element=None, unittext=None, **kwargs):
    self.__slots__ = ['len', 'offset', 'grp', 'text', 'unit_text']

    # initialization from kwargs
    if kwargs:
      for attrname in self.__slots__:
        setattr(self, attrname, kwargs.get(attrname, None))
    # initialization from an xml node
    else:
      self.len = e_element.attrib['len']
      self.offset = e_element.attrib['offset']
      self.grp = e_element.attrib['grp'] # if multiple groups, separated by |
      self.text = e_element.text.lower()

      self.unit_text = unittext 

  # for testing ambiguity
  def __hash__(self):
    return hash((self.len, self.offset, self.text))

  def __unicode__(self):
    to_string = lambda x: str(x) if x is not None else '' 
    return u'\t'.join(to_string(getattr(self, field)) for field in self.__slots__)

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

  def get_ambiguous_groups(self):
    groups = self.grp.split('|')
    return groups if len(groups) != 1 else None

def load_data(ssc_file_name):
  # loading XMLs
  parser = etree.XMLParser(encoding='utf-8')
  ssc = etree.parse(ssc_file_name, parser).getroot()

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

  return global_annotations