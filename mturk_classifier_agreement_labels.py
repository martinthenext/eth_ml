import sys
import codecs
import data 
from models import * 
import numpy as np
from sklearn import cross_validation
from scipy.sparse import coo_matrix


class OptionAwareNaiveBayesLeftRightMTurk(OptionAwareNaiveBayesLeftRight):

  def train(self, annotations, ylabels):
    X = self.vectorizer.fit_transform(annotations)
    self.classifier.partial_fit(X, ylabels, classes=range(10))


def get_mturk_classifier_agreement_label(mturk_vote_file_path, classifier_class, **kwargs):
  # train a classifier on ambiguous annotations
  ambig_annotations, labels = data.load_ambiguous_annotations_labeled(mturk_vote_file_path)

  classifier = classifier_class(**kwargs)
  classifier.train(ambig_annotations,labels)

  # classify annotations and output the agreement
  predicted_group_numbers = classifier.predict(ambig_annotations)
  voted_group_numbers = [data.Annotation.GROUP_MAPPING[label] for label in labels]
  agreement = [int(predicted == voted) for predicted, voted in zip(predicted_group_numbers, voted_group_numbers)]

  return np.mean(agreement)

#print get_mturk_classifier_agreement_label(sys.argv[1], OptionAwareNaiveBayesLeftRightMTurk)


class CountPrinter:
  def __init__(self, total):
    self.total = total
    self.current = 0

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
    voted_group_numbers = [data.Annotation.GROUP_MAPPING[label] for label in labels]
    agreement = [int(predicted == voted) for predicted, voted in zip(predicted_group_numbers, voted_group_numbers)]
    fold_errors.append(np.mean(agreement))
  
  return np.mean(fold_errors)

print crossvalidation(sys.argv[1], OptionAwareNaiveBayesLeftRightMTurk, n_folds=3, verbose = True)

  #return np.mean(agreement)