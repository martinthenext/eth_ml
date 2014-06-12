#!/usr/bin/env python

import argparse
import data
import models
from sklearn.externals import joblib
from lxml import etree
import sys
import codecs
import itertools

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)

def get_classifier_serialization_filename(classifier_name, ssc_file_list):
  corpus_codes = [sss_file_name.split('/')[-1].split("-")[0] for sss_file_name in ssc_file_list]
  return '%s_trained_on_%s' % (classifier_name, '_'.join(corpus_codes))

def write_to_log(unit_id, text, prob_dict):
  row = [unit_id, text] + list(itertools.chain.from_iterable(prob_dict.items()))
  sys.stderr.write(u'#AMBIG\t' + '\t'.join(row) + '\n')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train', nargs='+', type=str)
  parser.add_argument('-a', '--annotate', type=str)
  parser.add_argument('-o', '--output', type=str)
  args = parser.parse_args()

  annotations = []
  for ssc_file in args.train:
    print 'loading annotations'
    annotations += data.load_unambiguous_annotations(ssc_file)

  classifier = models.OptionAwareNaiveBayesLeftRightCutoff(window_size = 5, cutoff = 9)
  print 'training classifier'
  classifier.train(annotations)

  print 'processing probabilities'

  # SSC XML
  parser = etree.XMLParser(encoding='utf-8')
  tree = etree.parse(args.annotate, parser)
  ssc = tree.getroot()

  n_ambig_terms = 0

  for document in ssc.iter("document"):
    for unit in document.iter("unit"):
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

        write_to_log(unit.attrib['id'], to_classify.text, group_probabilities)

  sys.stderr.write('#TOTAL AMBIG TERMS\t' + str(n_ambig_terms))
  tree.write(args.output, xml_declaration=True, encoding='utf-8', pretty_print=True)
