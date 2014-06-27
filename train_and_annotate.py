#!/usr/bin/env python

'''

Train a classifier on unambiguous annotations from corpora and then annotate 
ambiguous cases from another corpus with group probabilities

If exclude_unit_dir is specified, units are excluded from training and all other units are 
deleted on the annotation step

Output corpus: stdout
Status messages: stderr

'''

import argparse
import data
import models
from sklearn.externals import joblib
from lxml import etree
import sys
import codecs
import itertools
from collections import Counter

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)

def get_classifier_serialization_filename(classifier_name, ssc_file_list):
  corpus_codes = [sss_file_name.split('/')[-1].split("-")[0] for sss_file_name in ssc_file_list]
  return '%s_trained_on_%s' % (classifier_name, '_'.join(corpus_codes))

def write_to_log(unit_id, text, prob_dict):
  row = [unit_id, text] + list(itertools.chain.from_iterable(prob_dict.items()))
  sys.stderr.write(u'#AMBIG\t' + '\t'.join(row) + '\n')

def get_unit_id_set(unit_id_file):
  result = set()
  with codecs.open(unit_id_file, 'r', 'utf-8') as f:
    for line in f:
      result.add(line[:-2])
  return result

def get_corpus_and_language(path):
  if "/" in path:
    path = path.split("/")[-1]

  slices = path.split("_")
  return slices[:2]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train', nargs='+', type=str)
  parser.add_argument('-a', '--annotate', type=str)
  parser.add_argument('-e', '--exclude-unit-dir', type=str, 
    help="Specify the exclude unit list folder with trailing slash if you want units excluded from training")
  parser.add_argument('-v', '--verbose', 
    help="Output disambiguation table to stdout", action="store_true")
  args = parser.parse_args()


  annotations = []
  for ssc_file in args.train:
    sys.stderr.write('loading annotations\n')
    # Load exlude unit IDs if any
    if args.exclude_unit_dir:
      # Figure out the filename of unit list
      coprus_and_lang = get_corpus_and_language(ssc_file)
      exclude_unit_path = args.exclude_unit_dir + "_".join(coprus_and_lang) + ".tsv"

      sys.stderr.write('reading ignore unit ids: %s\n' % exclude_unit_path)
      exclude_unit_ids = get_unit_id_set(exclude_unit_path)
    else:
      exclude_unit_ids = set()

    annotations += data.load_unambiguous_annotations(ssc_file, exclude_unit_ids)

  classifier = models.OptionAwareNaiveBayesLeftRightCutoff(window_size = 5, cutoff = 9)
  sys.stderr.write('training classifier\n')
  classifier.train(annotations)

  sys.stderr.write('processing probabilities\n')

  # If exclude folder is given, find a list of exclude units for the ANNOTATED
  # file and only leave those
  if args.exclude_unit_dir:
    coprus_and_lang = get_corpus_and_language(args.annotate)
    exclude_unit_path = args.exclude_unit_dir + "_".join(coprus_and_lang) + ".tsv"

    sys.stderr.write('annotating and deleting all units except for %s\n' % exclude_unit_path)
    units_to_leave = get_unit_id_set(exclude_unit_path)

  # Gathering statistics for ambiguous groups
  stats = Counter()

  # SSC XML
  parser = etree.XMLParser(encoding='utf-8')
  tree = etree.parse(args.annotate, parser)
  ssc = tree.getroot()

  n_ambig_terms = 0

  for document in ssc.iter("document"):
    for unit in document.iter("unit"):
      if args.exclude_unit_dir:
        # Delete all units if they are not EXCLUDE units
        if unit.attrib["id"] not in units_to_leave:
          unit.getparent().remove(unit)
          continue

      unit_text = unit.find("text").text

      non_empty_e = itertools.ifilter(lambda e: e.text is not None, unit.iter("e"))
      annotations = [data.Annotation(e, unit_text) for e in non_empty_e]

      conflicting_annotation_groups = [list(group) for key, group 
        in itertools.groupby(sorted(annotations, key=hash), key=hash)]      

      # Excluding groups of size 1 - they are not conflicting
      conflicting_annotation_groups = filter(lambda l: len(l) > 1, conflicting_annotation_groups)

      # Continue if the unit does not have conflicting annotations
      if not conflicting_annotation_groups:
        continue

      for conflicting_annotation_group in conflicting_annotation_groups:
        # Group the conflicting annotation list by 'grp'
        grp_key = lambda a: a.grp
        # Taking one element from group because they are similar except for CUIs
        ambiguous_annotations = [list(group)[0] for key, group
          in itertools.groupby(sorted(conflicting_annotation_group), key=grp_key)]

        if len(ambiguous_annotations) <= 1:
          continue

        n_ambig_terms += len(ambiguous_annotations)

        ambiguous_groups = [a.grp for a in ambiguous_annotations]
        statistics.update('|'.join(ambiguous_groups))

        # Getting probabilities from the classifier
        to_classify = ambiguous_annotations[0]
        X = classifier.vectorizer.transform([to_classify])
        probabilities = classifier.classifier.predict_proba(X)[0]

        group_probabilities = {}

        # Go through all conflicting_annotation_group list and add
        # probabilities of their 'grp' attributes
        for annotation in conflicting_annotation_group:
          # Find out the probability of its group
          group_index = data.Annotation.GROUP_MAPPING[annotation.grp]

          probability_str = "%.2f" % probabilities[group_index]
          annotation.e.attrib['grp_prob'] = probability_str

          group_probabilities[annotation.grp] = probability_str

        if args.verbose:
          write_to_log(unit.attrib['id'], to_classify.text, group_probabilities)

    if not len(document):
      document.getparent().remove(document)

  if args.verbose:
    sys.stderr.write('#TOTAL AMBIG TERMS\t' + str(n_ambig_terms) + '\n')

  # Ambiguous group statistics
  for group_stat in statistics.most_common():
    sys.stderr.write('#STAT_CASES %s\t%s' % group_stat)
  sys.stderr.write('#STAT_CASES Total\t%s' % sum(dict(c).values()))
 
  sys.stdout.write(etree.tostring(tree, pretty_print=True, encoding=unicode))
