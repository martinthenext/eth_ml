#!/usr/bin/env python

'''
Disambiguation: for every ambiguous case pick the group with higher probability if it is not less that cutoff
'''

import argparse
from lxml import etree
import itertools

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('annotated', help='SSC file annotated with class probabilities')
  parser.add_argument('output', help='Output file')
  parser.add_argument('-c', '--cutoff', help='Probability value that is enough to disambiguate, default is 0', default=0.0)
  args = parser.parse_args()

  # SSC XML
  parser = etree.XMLParser(encoding='utf-8')
  tree = etree.parse(args.annotated, parser)
  ssc = tree.getroot()

  # XPath to select all units with e elements with prob attribute
  for unit in tree.xpath('//unit[e[@grp_prob]]'):
    # Get e elements with grp_prob and group them by 
    # print 'BEFORE\n', etree.tostring(unit)
    annotated = list(unit.xpath('e[@grp_prob]'))
    # Grouping conflicting annotations together
    digest = lambda x: (x.attrib['len'], x.attrib['offset'], x.text)
    conflicting_groups = [list(l) for group, l in itertools.groupby(sorted(annotated, key=digest), key=digest)]
    for conflicting_annotations in conflicting_groups:
      # Select the biggest probability annotation
      prob = lambda x: float(x.attrib['grp_prob'])
      best_prob_annotation = sorted(conflicting_annotations, key=prob, reverse=True)[0]
      # If the biggest probability is smaller than the cutoff, don't disambiguate
      if prob(best_prob_annotation) < float(args.cutoff):
        continue
      # Delete all the <e> nodes from conflicting_annotations that are less probable that best_prob_annotation
      for annotation in conflicting_annotations:
        if prob(annotation) < prob(best_prob_annotation):
          annotation.getparent().remove(annotation)
    # print 'AFTER\n', etree.tostring(unit)

  tree.write(args.output, xml_declaration=True, encoding='utf-8', pretty_print=True)
