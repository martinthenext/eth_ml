''' Here classifiers live.

    Classifiers and Vectorizers here are wrappers around according
    sklearn objects that classify and vectorize annotations
'''

import random
import numpy
import operator
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import itertools

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

class ContextRestrictedBagOfWordsLeftRight(ContextRestrictedBagOfWords):
  def get_restricted_context_str_left(self, annotation):
    strings = map(lambda l: " ".join(l), self.get_restricted_context(annotation))
    return strings[0]

  def get_restricted_context_str_right(self, annotation):
    strings = map(lambda l: " ".join(l), self.get_restricted_context(annotation))
    return strings[1]

  def fit_transform(self, annotations):
    context_strings = [self.get_restricted_context_str(annotation)
     for annotation in annotations]
    # TODO should it just be 'fit'?
    # could be, but we do not need the output, as we do it separately for left and right
    self.vectorizer.fit_transform(context_strings)
    context_strings_left = [self.get_restricted_context_str_left(annotation)
     for annotation in annotations]
    context_strings_right = [self.get_restricted_context_str_right(annotation)
     for annotation in annotations]
    return sparse.hstack( (self.vectorizer.transform(context_strings_left),
      self.vectorizer.transform(context_strings_right)) )

  def transform(self, annotations):
    context_strings_left = [self.get_restricted_context_str_left(annotation)
     for annotation in annotations]
    context_strings_right = [self.get_restricted_context_str_right(annotation)
     for annotation in annotations]
    return sparse.hstack( (self.vectorizer.transform(context_strings_left),
      self.vectorizer.transform(context_strings_right) ))

class ContextRestrictedBagOfWordsLeftRightCutoff(ContextRestrictedBagOfWordsLeftRight):
  def __init__(self, window_size, min_df):
    self.vectorizer = CountVectorizer(min_df=min_df)
    self.window_size = window_size
    self.min_df = min_df

class ContextRestrictedBagOfWordsLeftRightStopWords(ContextRestrictedBagOfWordsLeftRight):
  def __init__(self, window_size):
    self.vectorizer = CountVectorizer(stop_words='english')
    self.window_size = window_size

class ContextRestrictedBagOfBigrams(ContextRestrictedBagOfWords):
  def __init__(self, window_size):
    self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,2))
    self.window_size = window_size

class NaiveBayesContextRestricted(AnnotationClassifier):
  def __init__(self, **kwargs):
    self.classifier = MultinomialNB()
    window_size = kwargs.get('window_size', 3)
    self.vectorizer = ContextRestrictedBagOfWords(window_size)

class NaiveBayesContextRestrictedLeftRight(AnnotationClassifier):
  def __init__(self, **kwargs):
    self.classifier = MultinomialNB()
    window_size = kwargs.get('window_size', 3)
    self.vectorizer = ContextRestrictedBagOfWordsLeftRight(window_size)

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
      max_probability = max(groups_probabilities, key=lambda x:x[1][0][1])
      predictions.append(Annotation.GROUP_MAPPING[str(max_probability[0])])
    return predictions

''' Randomly pick one option for group
'''
class OptionAwareRandom(AnnotationClassifier):
  def train(self, annotations):
    pass

  def get_random_group_code(self, annotation):
    group = random.choice(annotation.get_ambiguous_groups())
    return Annotation.GROUP_MAPPING[group]

  def predict(self, annotations):
    return [self.get_random_group_code(annotation) for annotation in annotations]

class OptionAwareNaiveBayes(NaiveBayesContextRestricted):
  def get_group_number(self, annotation, prob_vector):
    group_option_indices = annotation.get_group_number()
    group_option_prob = [prob_vector[group_option_index] for group_option_index in group_option_indices]
    group_index, _ = max(zip(group_option_indices, group_option_prob), key = lambda (index, prob): prob)
    return group_index

  def predict(self, annotations):
    X = self.vectorizer.transform(annotations)
    probs = self.classifier.predict_proba(X) #[n_samples, n_classes]
    return numpy.array([self.get_group_number(annotation, row)
     for row, annotation in itertools.izip(probs, annotations)])

class OptionAwareNaiveBayesLeftRight(OptionAwareNaiveBayes):
  def __init__(self, **kwargs):
    self.classifier = MultinomialNB()
    window_size = kwargs.get('window_size', 3)
    self.vectorizer = ContextRestrictedBagOfWordsLeftRight(window_size)

class OptionAwareNaiveBayesLeftRightCutoff(OptionAwareNaiveBayes):
  def __init__(self, **kwargs):
    self.classifier = MultinomialNB()
    window_size = kwargs.get('window_size', 3)
    cutoff = kwargs.get('cutoff', 3)
    self.vectorizer = ContextRestrictedBagOfWordsLeftRightCutoff(window_size, cutoff)

class OptionAwareNaiveBayesLeftRightStopWords(OptionAwareNaiveBayes):
  def __init__(self, **kwargs):
    self.classifier = MultinomialNB()
    window_size = kwargs.get('window_size', 3)
    self.vectorizer = ContextRestrictedBagOfWordsLeftRightStopWords(window_size)

class OptionAwareNaiveBayesBigrams(OptionAwareNaiveBayes):
  def __init__(self, **kwargs):
    self.classifier = MultinomialNB()
    window_size = kwargs.get('window_size', 8)
    self.vectorizer = ContextRestrictedBagOfBigrams(window_size)

class OptionAwareNaiveBayesFullContext(OptionAwareNaiveBayes):
  def __init__(self, **kwargs):
    self.classifier = MultinomialNB()
    self.vectorizer = FullContextVectorizer()

class FullContextBagOfWordsLeftRight(FullContextVectorizer):
  def get_left_right_full_tuple(self, annotation):
    left_context, _, right_context = annotation.get_slices()
    full_context = left_context + right_context
    return (left_context, right_context, full_context)

  def fit_transform(self, annotations, fit=True):
    side_tuples = [self.get_left_right_full_tuple(a) for a in annotations]
    left_contexts, right_contexts, full_contexts = zip(*side_tuples)
    if fit:
      self.vectorizer.fit(full_contexts)
    X_left = self.vectorizer.transform(left_contexts)
    X_right = self.vectorizer.transform(right_contexts)
    return sparse.hstack((X_left, X_right))

  def transform(self, annotations):
    return self.fit_transform(annotations, fit=False)

class OptionAwareNaiveBayesFullContextLeftRight(OptionAwareNaiveBayes):
  def __init__(self, **kwargs):
    self.classifier = MultinomialNB()
    self.vectorizer = FullContextBagOfWordsLeftRight()
