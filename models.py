''' Here classifiers live.

    Classifiers and Vectorizers here are wrappers around according
    sklearn objects that classify and vectorize annotations
'''

import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

''' Wrapper aroung sklearn to vectorize Annotation objects
'''
class FullContextVectorizer(object):
  def __init__(self):
    self.vectorizer = CountVectorizer()

  def fit_transform(self, annotations):
    return self.vectorizer.fit_transform(
      [annotation.get_context_string() for annotation in annotations])

  def transform(self, annotations):
    return self.vectorizer.transform(
      [annotation.get_context_string() for annotation in annotations])

''' Annotation classifier base class. Subclass it and specify
    a vectorizer and a classifiers
'''
class AnnotationClassifier(object):
  def train(self, annotations):
    X = self.vectorizer.fit_transform(annotations)
    y = numpy.array([annotation.get_group_number() for annotation in annotations])

    self.classifier.fit(X, y)

  def predict(self, annotations):
    X = self.vectorizer.transform(annotations)
    return self.classifier.predict(X)

class VeryVeryNaiveBayes(AnnotationClassifier):
  def __init__(self):
    self.classifier = MultinomialNB()
    self.vectorizer = FullContextVectorizer()

class ContextRestrictedBagOfWords(object):
  def __init__(self, window_size):
    self.vectorizer = CountVectorizer()
    self.window_size = window_size

  ''' On the withdrawal of the marketing [[application]] for Advexin
      -> ([u'of', u'the', u'marketing'], [u'for', u'Advexin'])
  '''
  def get_restricted_context(self, annotation):
    context_left, _, context_right = annotation.get_slices()
    words_left = context_left.strip().split(u" ")
    words_right = context_right.strip().split(u" ")

    return (words_left[-self.window_size:], words_right[:self.window_size])

  ''' On the withdrawal of the marketing [[application]] for Advexin
      -> of the marketing for Advexin
  '''
  def get_restricted_context_str(self, annotation):
    strings = map(lambda l: " ".join(l), self.get_restricted_context(annotation))
    return "%s %s" % tuple(strings)

  def fit_transform(self, annotations):
    context_strings = [self.get_restricted_context_str(annotation)
     for annotation in annotations]
    return self.vectorizer.fit_transform(context_strings)

  def transform(self, annotations):
    context_strings = [self.get_restricted_context_str(annotation)
     for annotation in annotations]
    return self.vectorizer.transform(context_strings)

class NaiveBayesContextRestricted(AnnotationClassifier):
  def __init__(self, **kwargs):
    self.classifier = MultinomialNB()
    window_size = kwargs.get('window_size', 3)
    self.vectorizer = ContextRestrictedBagOfWords(window_size)