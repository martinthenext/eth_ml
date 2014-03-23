''' Here classifiers live
    They train on Annotation objects and predict group codes
'''

import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class VeryVeryNaiveBayes(object):
  def __init__(self):
    self.classifier = MultinomialNB()
    self.vectorizer = CountVectorizer()

  def train(self, annotations):
    X = self.vectorizer.fit_transform([annotation.get_context_string() for annotation in annotations])
    y = numpy.array([annotation.get_group_number() for annotation in annotations])

    self.classifier.fit(X, y)

  def predict(self, annotations):
    X = self.vectorizer.transform([annotation.get_context_string() for annotation in annotations])
    return self.classifier.predict(X)