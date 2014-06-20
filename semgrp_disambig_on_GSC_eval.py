#!/usr/bin/env python

import argparse
import subprocess
import codecs
import sys

TEMP_FILE_NAME = 'tmp_semgrp_disambig.xml'

sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)
sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--training-corpora', nargs='+', type=str)
  parser.add_argument('-e', '--exclude-unit-dir', type=str, 
    help="Specify the exclude unit list folder with trailing slash if you want units excluded from training")
  parser.add_argument('-a', '--annotation-corpus', type=str)
  parser.add_argument('-c', '--cutoff')
  args = parser.parse_args()

  # Run train_and_annotate.py
  train_and_annotate_args = ['python', 'train_and_annotate.py', '-t'] + args.training_corpora + ['-a', args.annotation_corpus, '-e', args.exclude_unit_dir]
  proc = subprocess.Popen(train_and_annotate_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  sys.stderr.write('Training a classifier and annotating ambiguous cases\n')
  out, err = proc.communicate()

  # Run disambiguate_annotated.py
  disambiguate_annotated_args = ['python', 'disambiguate_annotated.py', '-c', args.cutoff]
  proc = subprocess.Popen(disambiguate_annotated_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  sys.stderr.write('Disambiguation\n')
  out, err = proc.communicate(input=out)

  # Write result to file
  sys.stderr.write('Writing result to temp file\n')
  with codecs.open(TEMP_FILE_NAME, 'w', 'utf-8') as f:
    f.write(out.decode('utf-8'))

  # Run centroid
  centroid_args = u" ".join(['java', '-cp', '~/lib/MantraAlign.jar', 'utils.Align', '-c', '~/lib/c1-cui-best.properties', TEMP_FILE_NAME, TEMP_FILE_NAME])

  proc = subprocess.Popen(centroid_args, stdout=subprocess.PIPE, shell=True)
  sys.stderr.write('Centroid\n')
  out, err = proc.communicate()

  sys.stdout.write(out.decode('utf-8'))
  
  # Run evaluator
  