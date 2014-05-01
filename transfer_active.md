# Transfer learning and Active learning

We use two datasets for training an annotation classifier:

1. Source dataset - unambiguous annotations from corpora
2. Target dataset - ambiguous annotations labeled by Mechanical Turk majority vote with 75% agreement threshold ([data](https://kitt.cl.uzh.ch/kitt/mantracrowd/disambig/vote_results.csv?AgreementThr=0.6))

*Note:* Mechanical Turk dataset is bigger than the one used in [results.org](results.org) - results not immediately comparable.

## WeightedPartialFitPassiveTransferClassifier

This classifier first trains MultinomialNB with source dataset, then uses `partial_fit` to train it also on the target dataset with higher `sample_weight`, see [source](transfer.py) and [docs](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.partial_fit). In the experiment for **1000 simulations**:

1. Split ambiguous annotation dataset into train (2/3) and validation (1/3) sets
1. Validate the initial transfer classifier (trained on unambiguous annotations only)
1. Target train it on all the train set 
1. Validate it again
1. Compute increase in agreement

For features, `FullContextBagOfWordsLeftRightCutoff(9)` vectorizer was used.

Then average in agreement increase is measured. 

### Average increase in agreement on validation set

|Source weight/Target weight|Source = EMEA|Source = Medline|
| --- | --- | --- |
|1/10|0%|1%|
|1/100|1%|1%|
|1/500|5%|0%|
|1/1000|5%|0%|
|1/5000|3%|1%|
|1/10000|-1%|-1%|
|1/50000|-3%|-1%|

*Results seem to have high variability*, so rounded to percents only.

## Passive learner

First results: accuracy progression with passive learner training the above classifier on EMEA and Medline corpora:

### EMEA

Accuracy progression:

    0.60 0.60 0.60 0.62 0.62 0.62 0.62 0.62 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.62 0.62 0.62 0.62 0.62 0.62 0.62 0.62 0.62 0.62 0.62 0.60 0.60 0.60 0.58 0.58 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.62 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.60 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.60 0.60 0.60 0.60 0.62 0.62 0.58 0.58 0.60 0.60 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.58 0.60 0.62 0.62 0.62 0.62 0.62 0.62 0.62 

Diffs:

    0.00 0.00 0.02 0.00 0.00 0.00 0.00 -0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 -0.02 0.00 0.00 -0.02 0.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 -0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 -0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.00 0.00 0.00 0.02 0.00 -0.04 0.00 0.02 0.00 -0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.02 0.00 0.00 0.00 0.00 0.00 0.00 

### Medline

Accuracy:

    0.73 0.73 0.75 0.75 0.75 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.73 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.69 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 0.71 

Diffs:

    0.00 0.02 0.00 0.00 -0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 -0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 -0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 
