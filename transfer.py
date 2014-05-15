from data import Annotation
from models import FullContextBagOfWordsLeftRightCutoff
import itertools
import random
from sklearn.naive_bayes import MultinomialNB
import numpy

'''

train_source - train a MultinomialNB (with weight 1)
train_target_online - partial_fit on it with weight `target_weight` > 1

'''
class WeightedPartialFitPassiveTransferClassifier(object):
  def __init__(self, target_weight):
    self.classifier = MultinomialNB()
    self.target_weight = target_weight
    self.vectorizer = FullContextBagOfWordsLeftRightCutoff(9)

  # Train on unambiguous annotatios which have a group number
  def train_source(self, annotations):
    X = self.vectorizer.fit_transform(annotations)
    y = numpy.array([annotation.get_group_number() for annotation in annotations])

    self.classifier.fit(X, y)

  # Train on ambiguous annotations with according group labels
  def train_target_online(self, annotations, labels):
    X = self.vectorizer.transform(annotations)
    y = numpy.array([Annotation.GROUP_MAPPING[label] for label in labels])

    weight_vector = [self.target_weight] * len(annotations)
    self.classifier.partial_fit(X, y, Annotation.GROUP_MAPPING.values(), weight_vector)

  def get_group_number_prob_pair(self, annotation, prob_vector):
    group_option_indices = annotation.get_group_number()
    group_option_prob = [prob_vector[group_option_index] for group_option_index in group_option_indices]
    return max(zip(group_option_indices, group_option_prob), key = lambda (index, prob): prob)
 
  def get_group_number(self, annotation, prob_vector):
    group_index, _ = self.get_group_number_prob_pair(annotation, prob_vector)
    return group_index

  # tested, results for the classifier trained on source are not random
  def predict(self, annotations):
    X = self.vectorizer.transform(annotations)
    probs = self.classifier.predict_proba(X) # [n_samples, n_classes]
    return numpy.array([self.get_group_number(annotation, row)
     for row, annotation in itertools.izip(probs, annotations)])

  # tested, results for the classifier trained on source are not random
  def get_max_probability(self, annotation, prob_vector):
    _, prob = self.get_group_number_prob_pair(annotation, prob_vector)
    return prob

  def get_prob_estimates(self, annotations):
    X = self.vectorizer.transform(annotations)
    probs = self.classifier.predict_proba(X)
    return numpy.array([self.get_max_probability(annotation, row)
      for row, annotation in itertools.izip(probs, annotations)])

# New classifier optimal on Medline
class WeightedPartialFitPassiveTransferClassifier2(WeightedPartialFitPassiveTransferClassifier):
  def __init__(self, target_weight):
    self.classifier = MultinomialNB()
    self.target_weight = target_weight
    self.vectorizer = ContextRestrictedBagOfWordsLeftRightCutoff(5, 9)

'''
TODO:

# Training

This classifier has two classifiers inside:

1. Source classifier, trained on unambiguous annotations
2. Target classifier, partially fitted on ambigous annotations from the pool

# Prediction

Probabilities for the resulting prediction are formed as follows:

P* = lambda * P1 + (1 - lambda) * P2

then the available group option with the highest probability is chosen

Target classifier should have the same vectorizer as source one, otherwize it
would be impossible to form a proper dictionary

'''