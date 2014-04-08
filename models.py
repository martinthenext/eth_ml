''' Here classifiers live.

    Classifiers and Vectorizers here are wrappers around according
    sklearn objects that classify and vectorize annotations
'''

import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from data import Annotation

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

class ContextRestrictedBagOfBigrams(ContextRestrictedBagOfWords):
  def __init__(self, window_size):
    self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(2,2))
    self.window_size = window_size

class NaiveBayesContextRestricted(AnnotationClassifier):
  def __init__(self, **kwargs):
    self.classifier = MultinomialNB()
    window_size = kwargs.get('window_size', 3)
    self.vectorizer = ContextRestrictedBagOfWords(window_size)

class NaiveBayesContexRestrictedBigrams(AnnotationClassifier):
  def __init__(self, **kwargs):
    self.classifier = MultinomialNB()
    window_size = kwargs.get('window_size', 3)
    self.vectorizer = ContextRestrictedBagOfBigrams(window_size)

''' 10 logistic regressions - one per get_group
    for every instance determine the probability of allowed groups
    (options for ambiguous term) and choose the one with highest probability
'''
class OptionAwareLogisticRegression(AnnotationClassifier):

  def __init__(self, **kwargs):
    self.classifiers = dict( (group, LogisticRegression())
     for group in Annotation.GROUP_NAMES)
    window_size = kwargs.get('window_size', 3)
    print window_size
    self.vectorizer = ContextRestrictedBagOfWords(window_size)

  def train(self, annotations):
    X = self.vectorizer.fit_transform(annotations)

    for groupindex in Annotation.GROUP_NAMES:
      current_group = Annotation.GROUP_MAPPING[groupindex]
      ylabels = numpy.array(map(lambda x: int(x.get_group_number() == current_group), annotations))
      self.classifiers[groupindex].fit(X, ylabels)

  def predict(self, annotations):
    X = self.vectorizer.transform(annotations)
    predictions=[]
    for index in range(len(annotations)):
      probabilities_all_groups = numpy.array(map(lambda group: self.classifiers[str(group)].predict_proba(X[index]),numpy.array(annotations[index].get_ambiguous_groups())))
      groups_probabilities = zip(annotations[index].get_ambiguous_groups(),probabilities_all_groups)
      max_probability = max(groups_probabilities)
      predictions.append(Annotation.GROUP_MAPPING[str(max_probability[0])])
    return predictions
