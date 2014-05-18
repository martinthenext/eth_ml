# Transfer and active learning with SVMs

Using SVM classifier on bag of words with frequency cutoff = 9, window of size 5, same as for [Naive Bayes](transfer_active.md).

### Logistic loss

|Target Weight| Accuracy Before | Accuracy After | Average gain|
| --- | --- | --- | --- |
|10|47.8|48.8|-0.009|
|100|47.8|54.5|-0.06|
|500|47.8|57.9|-0.1|
|1000|47.8|57.9|-0.1|

Using SVM classifier on bag of words with cutoff = 9 and using modified_huber. 

**Weight 1000**

![](http://davtyan.org/pml/WeightedSVMPartialFitPassiveTransferClassifier_Medline_weight1000.png)


**Weight 100**

![](http://davtyan.org/pml/WeightedSVMPartialFitPassiveTransferClassifier_Medline_weight100.png)


**Weight 10**

![](http://davtyan.org/pml/WeightedSVMPartialFitPassiveTransferClassifier_Medline_weight10.png)

### Modified Huber loss

|Target Weight| Accuracy Before | Accuracy After | Average gain|
| --- | --- | --- | --- |
|10|55.6|57.9|-0.02|
|100|55.6|56.2|-0.006|
|500|55.6|53.6|0.02|
|1000|55.6|52.7|0.028|

**Weight 1000**

![](http://davtyan.org/pml/WeightedSVMHuberPartialFitPassiveTransferClassifier_Medline_weight1000.png)

**Weight 100**

![](http://davtyan.org/pml/WeightedSVMHuberPartialFitPassiveTransferClassifier_Medline_weight100.png)

**Weight 10**

![](http://davtyan.org/pml/WeightedSVMHuberPartialFitPassiveTransferClassifier_Medline_weight10.png)
