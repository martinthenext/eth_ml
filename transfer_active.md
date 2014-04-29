# Transfer learning and Active learning

We use two datasets for training an annotation classifier:

1. Source dataset - unambiguous annotations from corpora
2. Target dataset - ambiguous annotations labeled by Mechanical Turk majority vote with 75% agreement threshold ([data](https://kitt.cl.uzh.ch/kitt/mantracrowd/disambig/vote_results.csv?AgreementThr=0.6))

*Note:* Mechanical Turk dataset is bigger than the one used in [results.org](results.org) - results not immediately comparable.

## WeightedPartialFitPassiveTransferClassifier

This classifier first trains MultinomialNB with source dataset, then uses `partial_fit` to train it also on the target dataset with higher `sample_weight`, see [source](transfer.py) and [docs](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.partial_fit). In the experiment for 100 simulations:

1. Split ambiguous annotation dataset into train (2/3) and validation (1/3) sets
1. Validate the initial transfer classifier (trained on unambiguous annotations only)
1. Target train it on all the train set 
1. Validate it again
1. Compute increase in agreement

Then average in agreement increase is measured. 

### Average increase in agreement on validation set

|Source weight/Target weight|Source = EMEA|Source = Medline|
| --- | --- | --- |
|1/100|1.2%|1.4%|
|1/1000|1%|4.6%|
|1/10000|-1.7%|-1.3%|

*Results seem to have high variability.*