#!/usr/bin/env python

import argparse
import subprocess
import codecs
import sys
from train_and_annotate import get_corpus_and_language

TEMP_FILE_NAME = 'tmp_semgrp_disambig.xml'
CENTROID_PROPERTIES_FILE = '~/lib/c1-cui-best.properties'
GOLD_CENTROID_DIR = '/data/clmantra/reannotation.nobackup/man-gsc/nonen-gsc-c1-2014-05-12.d/'

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

  # Getting corpus type and language of annotated corpus for later use
  corpus, language = get_corpus_and_language(args.annotation_corpus)

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
  centroid_args = u" ".join(['java', '-cp', '~/lib/MantraAlign.jar', 'utils.Align', '-c', CENTROID_PROPERTIES_FILE, TEMP_FILE_NAME, TEMP_FILE_NAME])

  proc = subprocess.Popen(centroid_args, stdout=subprocess.PIPE, shell=True)
  sys.stderr.write('Centroid\n')
  out, err = proc.communicate()

  with codecs.open(TEMP_FILE_NAME + '_cent', 'w', 'utf-8') as f:
    f.write(out.decode('utf-8'))

  # Tested: works with 
  # java -cp ~/lib/MantraAlign.jar utils.Evaluate  -c ~/lib/c1-cui-best.properties /data/clmantra/reannotation.nobackup/man-gsc/nonen-gsc-c1-2014-05-12.d/EMEA_de_man.xml <OUT>

  # Figure out where the Gold Standard centroid for the annotated corpus lives
  gold_centroid_path = GOLD_CENTROID_DIR + corpus + '_' + language + '_man.xml'
  sys.stderr.write(gold_centroid_path)
