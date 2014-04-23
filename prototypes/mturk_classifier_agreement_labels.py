import sys
import codecs
import data 
from models import * 
import numpy as np
from sklearn import cross_validation

class OptionAwareNaiveBayesLeftRightMTurk(OptionAwareNaiveBayesLeftRight):
  CLASS_PRIOR_MEDLINE = [
    0.12, 0.10, 0.03,
    0.27, 0.02, 0.13, 
    0.06, 0.23, 0.02, 
    0.04
  ]

  CLASS_PRIOR_EMEA =  [
    0.07, 0.29, 0.03,
    0.19, 0.02, 0.14, 
    0.05, 0.15, 0.02, 
    0.04
  ]

  def __init__(self, **kwargs):
    self.classifier = MultinomialNB(class_prior=self.CLASS_PRIOR_EMEA)
    window_size = kwargs.get('window_size', 3)
    self.vectorizer = ContextRestrictedBagOfWordsLeftRight(window_size)

  def train(self, annotations, ylabels):
    X = self.vectorizer.fit_transform(annotations)
    self.classifier.partial_fit(X, ylabels, classes=range(10))

class CountPrinter:
  def __init__(self, total):
    self.total = total
    self.current = 1

  def count(self):
    print '%s/%s' % (self.current, self.total)
    self.current += 1

def crossvalidation(mturk_vote_file_path, classifier_class, n_folds = 2, verbose=False, **kwargs):
  # train a classifier on ambiguous annotations
  ambig_annotations, labels = data.load_ambiguous_annotations_labeled(mturk_vote_file_path)
  ambig_annotations = np.array(ambig_annotations)
  labels = np.array(labels)

  folds = cross_validation.KFold(len(ambig_annotations), n_folds=n_folds, indices=True)

  counter = CountPrinter(n_folds)
  fold_errors = []

  for train_indices, test_indices in folds:
    if verbose: counter.count()
    classifier = classifier_class(**kwargs)
    classifier.train(ambig_annotations[train_indices], labels[train_indices])
    predicted_group_numbers = classifier.predict(ambig_annotations[test_indices])
    voted_group_numbers = [data.Annotation.GROUP_MAPPING[label] for label in labels[test_indices]]
    agreement = [int(predicted == voted) for predicted, voted in zip(predicted_group_numbers, voted_group_numbers)]
    fold_errors.append(np.mean(agreement))
  
  return np.mean(fold_errors)

print crossvalidation(sys.argv[1], OptionAwareNaiveBayesLeftRightMTurk, n_folds=105, verbose = False, window_size=2)
# If you want the nasty error to go away, edit the following file: ../sklearn\preprocessing\label.py and use int instead of bool :)