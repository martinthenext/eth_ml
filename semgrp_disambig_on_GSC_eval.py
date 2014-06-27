#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import codecs
import sys
from train_and_annotate import get_corpus_and_language
import datetime

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

  # Setting unique temp name
  TEMP_FILE_NAME = 'temp/' + str(datetime.datetime.now().time())

  # Run train_and_annotate.py
  train_and_annotate_args = ['python', 'train_and_annotate.py', '-t'] + args.training_corpora + ['-a', args.annotation_corpus, '-e', args.exclude_unit_dir, '-s']
  proc = subprocess.Popen(train_and_annotate_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  sys.stderr.write('#STATUS Training a classifier and annotating ambiguous cases\n')
  out, err = proc.communicate()
  sys.stderr.write(err)

  # Run disambiguate_annotated.py
  disambiguate_annotated_args = ['python', 'disambiguate_annotated.py', '-c', args.cutoff]
  proc = subprocess.Popen(" ".join(disambiguate_annotated_args), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  sys.stderr.write('#STATUS Disambiguation\n')
  out, err = proc.communicate(input=out)

  # Write result to file
  sys.stderr.write('#STATUS Writing result to temp file\n')
  disambig_filename = TEMP_FILE_NAME + '.xml'
  with codecs.open(disambig_filename, 'w', 'utf-8') as f:
    f.write(out.decode('utf-8'))

  # Figure out where the Gold Standard centroid for the annotated corpus lives
  gold_centroid_path = GOLD_CENTROID_DIR + corpus + '_' + language + '_man.xml'
  sys.stderr.write('#STATUS Gold centroid at %s\n' % gold_centroid_path)

  # Run evaluate
  evaluate_call = 'java -cp ~/lib/MantraAlign.jar utils.Evaluate -c %s %s %s' % (CENTROID_PROPERTIES_FILE, gold_centroid_path, disambig_filename)
  proc = subprocess.Popen(evaluate_call, stdout=subprocess.PIPE, shell=True)
  sys.stderr.write('#STATUS Evaluate\n')
  out, err = proc.communicate()

  temp_eval_filename = TEMP_FILE_NAME + '_eval.xml'
  with codecs.open(temp_eval_filename, 'w', 'utf-8') as f:
    f.write(out.decode('utf-8'))

  # Run summary
  temp_summary_filename = TEMP_FILE_NAME + '_summ.html'
  summary_call = 'java -cp ~/lib/MantraAlign.jar  utils.SummaryStats %s %s' % (temp_eval_filename, temp_summary_filename)
  proc = subprocess.Popen(summary_call, stdout=subprocess.PIPE, shell=True)
  sys.stderr.write('#STATUS Summary\n')
  out, err = proc.communicate()

  # Run html2text tool on summary
  html2text_call = 'html2text %s' % temp_summary_filename
  sys.stderr.write(html2text_call + '\n')
  proc = subprocess.Popen(html2text_call, stdout=subprocess.PIPE, shell=True)
  sys.stderr.write('#STATUS HTML2TEXT\n')
  out, err = proc.communicate()

  # Run perl script
  
  '''
  perl_call = u"""perl -ln -e 'BEGIN{$info="$*"; $info =~ s/(Patent|Medline|EMEA)_(.+?)_(.+?)$/GSC\t$3\t$1\t$2/;}' -e 'next if /^(GRP|<\?xml)/;s/\s+/\t/g; s/^TOTAl/TOTAL/; $$_ = "$info\t" . $_;print;' | perl -lne 's/ï¿½?/0.0/g;s/ï¿/0.0/g;print unless /^(\S+)\t\1/;'"""
  proc = subprocess.Popen(perl_call, stdout=subprocess.PIPE, shell=True)
  sys.stderr.write('#Summary formatting\n')
  out, err = proc.communicate(input=out)
  '''
  
  sys.stdout.write(out.decode('utf-8'))
