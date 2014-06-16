#!/usr/bin/env python

import argparse
from lxml import etree

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('annotated', help='SSC file annotated with class probabilities')
  parser.add_argument('output', help='Output file')
  parser.add_argument('-c', '--cutoff', help='Probability value that is enough to disambiguate, default is 0', default=0)
  args = parser.parse_args()

  # SSC XML
  parser = etree.XMLParser(encoding='utf-8')
  tree = etree.parse(args.annotated, parser)
  ssc = tree.getroot()

  # XPath to select all e elements with prob attribute
  for e in tree.xpath('//e[@grp_prob]'):
    print e.attrib['grp_prob']