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

  # Train on ambiguous annotations with according group number labels
  def train_target_online(self, annotations, labels):
    X = self.vectorizer.transform(annotations)
    y = numpy.array(labels)

    weight_vector = [self.target_weight] * len(annotations)
    self.classifier.partial_fit(X, y, Annotation.GROUP_MAPPING.values(), weight_vector)

  def get_group_number(self, annotation, prob_vector):
    group_option_indices = annotation.get_group_number()
    group_option_prob = [prob_vector[group_option_index] for group_option_index in group_option_indices]
    group_index, _ = max(zip(group_option_indices, group_option_prob), key = lambda (index, prob): prob)
    return group_index

  def predict(self, annotations):
    X = self.vectorizer.transform(annotations)
    probs = self.classifier.predict_proba(X) # [n_samples, n_classes]
    return numpy.array([self.get_group_number(annotation, row)
     for row, annotation in itertools.izip(probs, annotations)])

  # Passive learner
  def pick_and_train(annotation_label_tuple_pool, sample_size):
    annotation_label_tuples = random.sample(annotation_label_tuple_pool, sample_size)
    annotations, labels = zip(*annotation_label_tuples)
    self.train_target_online(annotations, labels)

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