from data import FullContextBagOfWordsLeftRightCutoff

'''

# Training

This classifier has two classifiers inside:

1. Source classifier, trained on unambiguous annotations
2. Targer classifier, partially fitted on ambigous annotations from the pool

# Prediction

Probabilities for the resulting prediction are formed as follows:

P* = lambda * P1 + (1 - lambda) * P2

then the available group option with the highest probability is chosen

Target classifier should have the same vectorizer as source one, otherwize it
would be impossible to form a proper dictionary

'''

class InductiveTransferPassiveClassifier(object):
  def __init__(self):
    self.source_classifier = MultinomialNB()
    self.target_classifier = MultinomialNB()
    self.vectorizer = FullContextBagOfWordsLeftRightCutoff(9)

  # Train on unambiguous annotatios which have a group number
  def train_source(self, annotations):
    pass

  # Train on ambiguous annotations with according group number labels
  def train_target_online(self, annotations, labels):
    pass