#!/usr/bin/env python

import argparse
import subprocess
import codecs
import sys

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

  train_and_annotate_args = ['python', 'train_and_annotate.py', '-t'] + args.training_corpora + ['-a', args.annotation_corpus, '-e', args.exclude_unit_dir]
  proc = subprocess.Popen(train_and_annotate_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out, err = proc.communicate()
  
  sys.stdout.write(out.decode('utf-8'))
