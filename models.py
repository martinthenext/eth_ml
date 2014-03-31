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
